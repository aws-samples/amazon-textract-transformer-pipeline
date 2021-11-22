# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Custom CloudFormation Resource for post-creation setup of a SageMaker Studio user

Clones a (public) 'GitRepository' into the user's home folder.

Updating or deleting this resource does not currently do anything. Errors in the setup process are
also ignored (typically don't want to roll back the whole stack just because we couldn't clone a
repo - as users can always do it manually!)
"""
# Python Built-Ins:
import logging
import os
import traceback

# External Dependencies:
import boto3
import cfnresponse
from git import Repo

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
    logging.info("**Setting up user")
    result = create_user_setup(resource_config)
    cfnresponse.send(
        event,
        context,
        cfnresponse.SUCCESS,
        {"UserProfileName": result["UserProfileName"]},
        physicalResourceId=result["UserProfileName"],
    )


def handle_delete(event, context):
    logging.info("**Received delete event")
    user_profile_name = event["PhysicalResourceId"]
    domain_id = event["ResourceProperties"]["DomainId"]
    logging.info("**Deleting user setup")
    delete_user_setup(domain_id, user_profile_name)
    cfnresponse.send(
        event,
        context,
        cfnresponse.SUCCESS,
        {},
        physicalResourceId=event["PhysicalResourceId"],
    )


def handle_update(event, context):
    logging.info("**Received update event")
    user_profile_name = event["PhysicalResourceId"]
    domain_id = event["ResourceProperties"]["DomainId"]
    git_repo = event["ResourceProperties"]["GitRepository"]
    logging.info("**Updating user setup")
    update_user_setup(domain_id, user_profile_name, git_repo)
    cfnresponse.send(
        event,
        context,
        cfnresponse.SUCCESS,
        {},
        physicalResourceId=event["PhysicalResourceId"],
    )


def chown_recursive(path, uid=-1, gid=-1):
    """Workaround for os.chown() not having a recursive option for folders"""
    for dirpath, dirnames, filenames in os.walk(path):
        os.chown(dirpath, uid, gid)
        for filename in filenames:
            os.chown(os.path.join(dirpath, filename), uid, gid)


def create_user_setup(config):
    domain_id = config["DomainId"]
    user_profile_name = config["UserProfileName"]
    git_repo = config["GitRepository"]
    efs_uid = config["HomeEfsFileSystemUid"]
    print(f"Setting up user: {config}")
    try:
        # The root of the EFS contains folders named for each user UID, but these may not be
        # created before the user has first logged in (could os.listdir("/mnt/efs") to check):
        print("Creating/checking home folder...")
        home_folder = f"/mnt/efs/{efs_uid}"
        os.makedirs(home_folder, exist_ok=True)
        # Set correct ownership permissions for this folder straight away, in case a later process
        # errors out
        os.chown(home_folder, int(efs_uid), -1)

        # Now ready to clone in Git content (or whatever else...)
        print(f"Cloning code... {git_repo}")
        # Our target folder for Repo.clone_from() needs to be the *actual* target folder, not the
        # parent under which a new folder will be created, so we'll infer that from the repo name:
        repo_folder_name = git_repo.rpartition("/")[2]
        if repo_folder_name.lower().endswith(".git"):
            repo_folder_name = repo_folder_name[: -len(".git")]
        Repo.clone_from(git_repo, f"{home_folder}/{repo_folder_name}")

        # Remember to set ownership/permissions for all the stuff we just created, to give the user
        # write access:
        chown_recursive(f"{home_folder}/{repo_folder_name}", uid=int(efs_uid))
        print("All done")
    except Exception as e:
        # Don't bring the entire CF stack down just because we couldn't copy a repo:
        print("IGNORING CONTENT SETUP ERROR")
        traceback.print_exc()

    logging.info("**SageMaker Studio user '%s' set up successfully", user_profile_name)
    return {"UserProfileName": user_profile_name}


def delete_user_setup(domain_id, user_profile_name):
    logging.info(
        "**Deleting user setup is a no-op: user '%s' on domain '%s",
        user_profile_name,
        domain_id,
    )
    return {"UserProfileName": user_profile_name}


def update_user_setup(domain_id, user_profile_name, git_repo):
    logging.info(
        "**Updating user setup is a no-op: user '%s' on domain '%s",
        user_profile_name,
        domain_id,
    )
    return {"UserProfileName": user_profile_name}
