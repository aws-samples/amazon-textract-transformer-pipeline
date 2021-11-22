# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Custom CloudFormation Resource for a SageMaker Studio Domain (with additional outputs)

As well as creating a SMStudio domain, this implementation:
- Defaults to the default VPC, or to any VPC when exactly one is present, if not explicitly
  configured
- Defaults to all default subnets if any are present, or else all subnets in VPC, if not
  explicitly set
- Discovers and outputs a list of security group IDs (default+SM-generated) that downstream
  resources may use to perform user setup actions on the Elastic File System
"""
# Python Built-Ins:
import logging
import time
import traceback

# External Dependencies:
import boto3
import cfnresponse

# Local Dependencies:
import vpctools

ec2 = boto3.client("ec2")
smclient = boto3.client("sagemaker")


def lambda_handler(event, context):
    try:
        request_type = event["RequestType"]
        if request_type == "Create":
            handle_create(event, context)
        elif request_type == "Update":
            handle_update(event, context)
        elif request_type == "Delete":
            handle_delete(event, context)
        else:
            cfnresponse.send(
                event,
                context,
                cfnresponse.FAILED,
                {},
                error=f"Unsupported CFN RequestType '{request_type}'",
            )
    except Exception as e:
        logging.error("Uncaught exception in CFN custom resource handler - reporting failure")
        traceback.print_exc()
        cfnresponse.send(
            event,
            context,
            cfnresponse.FAILED,
            {},
            error=str(e),
        )
        raise e


def handle_create(event, context):
    logging.info("**Received create request")
    resource_config = event["ResourceProperties"]

    # We split out pre- and post-processing because we'd like to always report our correct
    # physicalResourceId if erroring out after the actual creation, so that the subsequent deletion
    # request can clean up.
    logging.info("**Preparing studio domain creation parameters")
    create_domain_args = preprocess_create_domain_args(resource_config)
    logging.info("**Creating studio domain")
    creation = smclient.create_domain(**create_domain_args)
    _, _, domain_id = creation["DomainArn"].rpartition("/")
    try:
        result = post_domain_create(domain_id)
        domain_desc = result["DomainDescription"]
        response = {
            "DomainId": domain_desc["DomainId"],
            "DomainName": domain_desc["DomainName"],
            "HomeEfsFileSystemId": domain_desc["HomeEfsFileSystemId"],
            "SubnetIds": ",".join(domain_desc["SubnetIds"]),
            "Url": domain_desc["Url"],
            "VpcId": domain_desc["VpcId"],
            "ProposedAdminSubnetCidr": result["ProposedAdminSubnetCidr"],
            "InboundEFSSecurityGroupId": result["InboundEFSSecurityGroupId"],
            "OutboundEFSSecurityGroupId": result["OutboundEFSSecurityGroupId"],
        }
        print(response)
        cfnresponse.send(
            event,
            context,
            cfnresponse.SUCCESS,
            response,
            physicalResourceId=domain_id,
        )
    except Exception as e:
        logging.error("Uncaught exception in post-creation processing")
        traceback.print_exc()
        cfnresponse.send(
            event,
            context,
            cfnresponse.FAILED,
            {},
            physicalResourceId=domain_id,
            error=str(e),
        )


def handle_delete(event, context):
    logging.info("**Received delete event")
    domain_id = event["PhysicalResourceId"]
    try:
        smclient.describe_domain(DomainId=domain_id)
    except smclient.exceptions.ResourceNotFound as exception:
        # Already does not exist -> deletion success
        cfnresponse.send(
            event,
            context,
            cfnresponse.SUCCESS,
            {},
            physicalResourceId=event["PhysicalResourceId"],
        )
        return
    logging.info("**Deleting studio domain")
    delete_domain(domain_id)
    cfnresponse.send(
        event,
        context,
        cfnresponse.SUCCESS,
        {},
        physicalResourceId=event["PhysicalResourceId"],
    )


def handle_update(event, context):
    logging.info("**Received update event")
    domain_id = event["PhysicalResourceId"]
    default_user_settings = event["ResourceProperties"]["DefaultUserSettings"]
    logging.info("**Updating studio domain")
    update_domain(domain_id, default_user_settings)
    # TODO: Should we wait here for the domain to enter active state again?
    cfnresponse.send(
        event,
        context,
        cfnresponse.SUCCESS,
        {"DomainId": domain_id},
        physicalResourceId=event["PhysicalResourceId"],
    )


def preprocess_create_domain_args(config):
    default_user_settings = config["DefaultUserSettings"]
    domain_name = config["DomainName"]
    vpc_id = config.get("VPC")
    subnet_ids = config.get("SubnetIds")

    if not vpc_id:
        # Try to look up the default VPC ID:
        # TODO: NextToken handling on this list API?
        available_vpcs = ec2.describe_vpcs()["Vpcs"]
        if len(available_vpcs) <= 0:
            raise ValueError("No default VPC exists - cannot create SageMaker Studio Domain")

        default_vpcs = list(filter(lambda v: v["IsDefault"], available_vpcs))
        if len(default_vpcs) == 1:
            vpc = default_vpcs[0]
        elif len(default_vpcs) > 1:
            raise ValueError("'VPC' not specified in config, and multiple default VPCs found")
        else:
            if len(available_vpcs) == 1:
                vpc = available_vpcs[0]
                logging.warning(f"Found exactly one (non-default) VPC: Using {vpc['VpcId']}")
            else:
                raise ValueError(
                    "'VPC' not specified in config, and multiple VPCs found with no 'default' VPC"
                )
        vpc_id = vpc["VpcId"]

    if not subnet_ids:
        # Use all the subnets
        # TODO: NextToken handling on this list API?
        available_subnets = ec2.describe_subnets(
            Filters=[
                {
                    "Name": "vpc-id",
                    "Values": [vpc_id],
                }
            ],
        )["Subnets"]
        default_subnets = list(filter(lambda n: n["DefaultForAz"], available_subnets))
        subnet_ids = [
            n["SubnetId"]
            for n in (default_subnets if len(default_subnets) > 0 else available_subnets)
        ]
    elif isinstance(subnet_ids, str):
        subnet_ids = subnet_ids.split(",")

    return {
        "DomainName": domain_name,
        "AuthMode": "IAM",
        "DefaultUserSettings": default_user_settings,
        "SubnetIds": subnet_ids,
        "VpcId": vpc_id,
    }


def post_domain_create(domain_id):
    created = False
    time.sleep(0.2)
    while not created:
        description = smclient.describe_domain(DomainId=domain_id)
        status_lower = description["Status"].lower()
        if status_lower == "inservice":
            created = True
            break
        elif "fail" in status_lower:
            raise ValueError(f"Domain {domain_id} entered failed status")
        time.sleep(5)
    logging.info("**SageMaker domain created successfully: %s", domain_id)

    vpc_id = description["VpcId"]
    # Retrieve the VPC security groups set up by SageMaker for EFS communication:
    inbound_efs_sg_id, outbound_efs_sg_id = vpctools.get_studio_efs_security_group_ids(
        domain_id,
        vpc_id,
    )
    # Propose a valid subnet to create in this VPC for managing further setup actions:
    proposed_admin_subnet = vpctools.propose_subnet(vpc_id)
    return {
        "DomainDescription": description,
        "ProposedAdminSubnetCidr": proposed_admin_subnet["CidrBlock"],
        "InboundEFSSecurityGroupId": inbound_efs_sg_id,
        "OutboundEFSSecurityGroupId": outbound_efs_sg_id,
    }


def delete_domain(domain_id):
    response = smclient.delete_domain(
        DomainId=domain_id,
        RetentionPolicy={"HomeEfsFileSystem": "Delete"},
    )
    deleted = False
    time.sleep(0.2)
    while not deleted:
        try:
            smclient.describe_domain(DomainId=domain_id)
        except smclient.exceptions.ResourceNotFound:
            logging.info(f"Deleted domain {domain_id}")
            deleted = True
            break
        time.sleep(5)
    return response


def update_domain(domain_id, default_user_settings):
    response = smclient.update_domain(
        DomainId=domain_id,
        DefaultUserSettings=default_user_settings,
    )
    updated = False
    time.sleep(0.2)
    while not updated:
        response = smclient.describe_domain(DomainId=domain_id)
        if response["Status"] == "InService":
            updated = True
        else:
            logging.info("Updating domain %s.. %s", domain_id, response["Status"])
        time.sleep(5)
    return response
