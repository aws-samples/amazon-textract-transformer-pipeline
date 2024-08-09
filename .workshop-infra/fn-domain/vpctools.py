# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Utilities for analyzing VPCs for use with SageMaker Studio"""
# Python Built-Ins:
import ipaddress
from typing import List, Tuple, Union

# External Dependencies:
import boto3

ec2 = boto3.client("ec2")


def get_studio_efs_security_group_ids(
    studio_domain_id: str, vpc_id: str
) -> Tuple[Union[str, None], Union[str, None]]:
    """Retrieve the security groups you need for [inbound, outbound] comms with SMStudio EFS

    Returns
    -------
    inbound : Union[str, None]
        Security Group ID for inbound connection from SMStudio filesystem, or None if not found
    outbound : str
        Secrity Group ID for outbound connection to SMStudio filesystem, or None if not found

    Raises
    ------
    ValueError :
        If multiple potential SGs are found for either inbound or outbound connection (suggests
        duplication or otherwise erroneous SMStudio/VPC setup).
    Other :
        As per boto3 EC2 describe_security_groups()
    """
    inbound_sg_name = f"security-group-for-inbound-nfs-{studio_domain_id}"
    outbound_sg_name = f"security-group-for-outbound-nfs-{studio_domain_id}"
    nfs_sgs = ec2.describe_security_groups(
        Filters=[
            {"Name": "vpc-id", "Values": [vpc_id]},
            {"Name": "group-name", "Values": [inbound_sg_name, outbound_sg_name]},
        ],
    )["SecurityGroups"]
    inbound_sgs = list(
        filter(
            lambda sg: sg["GroupName"] == inbound_sg_name,
            nfs_sgs,
        )
    )
    n_inbound_sgs = len(inbound_sgs)
    outbound_sgs = list(
        filter(
            lambda sg: sg["GroupName"] == outbound_sg_name,
            nfs_sgs,
        )
    )
    n_outbound_sgs = len(outbound_sgs)
    if n_inbound_sgs > 1 or n_outbound_sgs > 1:
        raise ValueError(
            "Found duplicate EFS security groups for SMStudio {}: Got {} inbound, {} outbound".format(
                studio_domain_id,
                n_inbound_sgs,
                n_outbound_sgs,
            )
        )
    return (
        inbound_sgs[0]["GroupId"] if n_inbound_sgs else None,
        outbound_sgs[0]["GroupId"] if n_outbound_sgs else None,
    )


def propose_subnet(vpc_id, new_subnet_prefixlen=26):
    """Propose a valid configuration for a new IPv4 subnet to add to the VPC for CF stack purposes

    Parameters
    ----------
    vpc_id : str
        ID of the VPC to propose a subnet for
    new_subnet_prefixlen : int (optional)
        CIDR mask length in bits for requested new subnet to propose. Defaults to 26 bits (64 IPs)
    """

    # Get VPC info:
    vpc_list = ec2.describe_vpcs(
        Filters=[{"Name": "vpc-id", "Values": [vpc_id]}],
    )["Vpcs"]
    if not len(vpc_list):
        raise ValueError(f"VPC ID {vpc_id} not found")
    vpc_description = vpc_list[0]
    existing_subnets = ec2.describe_subnets(
        Filters=[{"Name": "vpc-id", "Values": [vpc_id]}],
    )["Subnets"]

    # Load CIDRs of provided VPC and existing subnets with Python ipaddress library:
    vpc_net = ipaddress.ip_network(vpc_description["CidrBlock"])
    existing_nets = list(
        map(
            lambda subnet: ipaddress.ip_network(subnet["CidrBlock"]),
            existing_subnets,
        )
    )

    # Validate existing configuration:
    # (Could probably skip this since we just retrieved fresh data, but might help to prevent any
    # weird errors manifesting as harder-to-interpret issues further down)
    for subnet in existing_nets:
        if not subnet.subnet_of(vpc_net):
            raise ValueError(f"Listed 'subnet' {subnet} is not inside VPC {vpc_net}")
        for checknet in existing_nets:
            if checknet != subnet and subnet.overlaps(checknet):
                raise ValueError(f"Listed subnets {subnet} and {checknet} overlap")

    # Calculate remaining vacant ranges:
    available_nets = [vpc_net]
    for subnet in existing_nets:
        next_available = []
        for vacancy in available_nets:
            if vacancy.subnet_of(subnet):
                # This gap is fully contained by `subnet`
                continue
            try:
                # Preserve the list of subranges in `vacancy` after excluding `subnet`:
                next_available += list(vacancy.address_exclude(subnet))
            except ValueError:
                # This `vacancy` does not contain `subnet`:
                next_available.append(vacancy)
        available_nets = next_available
    available_nets.sort()

    # Select the first available subnet of requested size:
    try:
        parent = next(
            filter(
                lambda n: n.prefixlen <= new_subnet_prefixlen,
                available_nets,
            )
        )
    except StopIteration:
        raise ValueError(f"No vacant subnets of requested size /{new_subnet_prefixlen} left in VPC")

    if parent.prefixlen == new_subnet_prefixlen:
        proposed_net = parent
    else:
        diff = new_subnet_prefixlen - parent.prefixlen
        proposed_net = next(parent.subnets(diff))

    return {"CidrBlock": str(proposed_net)}
