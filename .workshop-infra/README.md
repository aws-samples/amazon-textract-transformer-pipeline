# Infrastructure for SageMaker Workshop with a CDK solution stack

This folder provides a helper stack which will:

- Create a SageMaker Notebook Instance with the repository cloned in
- Create an (IAM-authenticated) SageMaker Studio domain, with a user profile, with the repository cloned in (and some VPC infrastructure required to make that happen)
- Run a one-off AWS CodeBuild build to download the repository, `poetry install` the dependencies and `cdk deploy --all` stacks in the solution

It's intended to help automate setting up workshops on temporary AWS accounts, with CDK-based solutions (like this one) that assume a SageMaker notebook environment will be provisioned separately.

## Prerequisites and Caveats

This helper stack assumes that (in your target AWS Region):

- You have not yet onboarded to SageMaker Studio
- You have a default VPC you're willing to use with standard configuration, or else would like to use a custom VPC but are comfortable checking the compatibility of the stack with your VPC configuration.

> ⚠️ This stack is oriented towards convenience of **getting started** and first exploring SageMaker Studio with the companion solution stack. It is **not recommended for long-lived environments**.
>
> In particular, **be aware that:**
>
> - The stack grants broad power user permissions to the CodeBuild job (for whatever resources the CDK deployment may need to create)
> - When you delete the stack
>    - The SageMaker Studio setup for your target AWS Region will be deleted (and the stack should *fail* to delete if you have any users running 'apps' in Studio apart from the ones set up by the stack. You can manage these through the [SageMaker console UI](https://console.aws.amazon.com/sagemaker/home?#/studio))
>    - The CDK solution deployed deployed by the CodeBuild project will *not* automatically be cleaned up

## Developing and Deploying Locally

In addition to having an AWS account, you'll need an environment with:

- The [AWS CLI](https://aws.amazon.com/cli/)
- The [AWS SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-install.html)
- A Docker-compatible container runtime such as [Docker Desktop](https://www.docker.com/products/docker-desktop)
- A `make` utility such as [GNU Make](https://www.gnu.org/software/make/) - probably already installed if you have some bundled build tools already.
- *Probably* a UNIX-like (non-Windows) shell if you want things to run smoothly... But you can always give it a try and resort to translating commands from the [Makefile](Makefile) if things go wrong.

You'll also need:

- Sufficient access (log in with `aws configure`) to be able to deploy the stacks in your target region
- An *[Amazon S3](https://s3.console.aws.amazon.com/s3/home) Bucket* to use for staging deployment assets (Lambda bundles, etc)

**Step 1: Build the Lambda bundles and final CloudFormation template to S3 with AWS SAM**

(This command builds your assets and CloudFormation template, and stages them to your nominated Amazon S3 bucket)

```sh
make package DEPLOYMENT_BUCKET_NAME=DOC-EXAMPLE-BUCKET
```

**Step 2: Deploy (create or update) the stack**

```sh
make deploy STACK_NAME=workshopstack
```

***Alternative: Build and create the stack in one go**

(This option only *creates* stacks, and disables rollback, for easier debugging)

```sh
make all DEPLOYMENT_BUCKET_NAME=example-bucket STACK_NAME=workshopstack
```

There's also a `make delete` option to help with cleaning up.

## Preparing Templates for Multi-Region Deployment

If you'd like your template to be deployable in multiple AWS Regions:

- Set up an asset hosting bucket in each region of interest, and use the AWS Region ID (e.g. `us-east-1`) in the bucket names
- Set up cross-region replication to copy contents from your lead region to other regions
- Run the `make package` script against your lead region

The generated template will be automatically post-processed (by [sam-postproc.py](sam-postproc.py)) to tokenize S3 references to hosted assets to refer to the `${AWS::Region}` placeholder.
