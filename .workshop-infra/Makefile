# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
STACK_NAME?="textract-transformers-workshop"
DEPLOYMENT_BUCKET_NAME?="UNDEFINED"
DEPLOYMENT_BUCKET_PREFIX?=""
TARGET_REPO?="https://github.com/aws-samples/amazon-textract-transformer-pipeline"

target:
	$(info ${HELP_MESSAGE})
	@exit 0

package: ##=> Build SAM template & assets to CloudFormation on S3
	$(info [*] Building AWS SAM stack...)
	sam build \
			--use-container \
			--template template.sam.yaml && \
		sam package \
			--s3-bucket $(DEPLOYMENT_BUCKET_NAME) \
			--s3-prefix $(DEPLOYMENT_BUCKET_PREFIX)sam \
			--use-json \
			--output-template-file template.tmp.json && \
		python sam-postproc.py template.tmp.json template.tmp.json && \
		aws s3 cp template.tmp.json \
			s3://$(DEPLOYMENT_BUCKET_NAME)/$(DEPLOYMENT_BUCKET_PREFIX)template.cf.json

# CF with --disable-rollback is faster for debugging than sam deploy
create: ##=> Create services stack (only)
	$(info [*] Deploying...)
	aws cloudformation create-stack \
			--template-body file://template.tmp.yaml \
			--stack-name $(STACK_NAME) \
			--capabilities CAPABILITY_IAM CAPABILITY_AUTO_EXPAND \
			--disable-rollback
			# --parameters \
			#   ParameterKey=ParamName,ParameterValue=$(PARAM_VAR)

deploy: ##=> Deploy services (flexible create or update)
	$(info [*] Deploying...)
	sam deploy \
			--template-file template.tmp.yaml \
			--stack-name $(STACK_NAME) \
			--capabilities CAPABILITY_IAM CAPABILITY_AUTO_EXPAND \
			--no-fail-on-empty-changeset
			# --parameter-overrides \
			# 	ParamName=$(PARAM_VAR)

all: ##=> Build and create stack
	@$(MAKE) package
	@$(MAKE) create

delete: ##=> Delete services
	$(info [*] Deleting stack...)
	aws cloudformation delete-stack --stack-name $(STACK_NAME)


#############
#  Helpers  #
#############

define HELP_MESSAGE

	STACK_NAME: "textract-transformers-workshop"
		Description: Stack Name to deploy/redeploy to
	DEPLOYMENT_BUCKET_NAME:
		Description: Amazon S3 bucket for staging built SAM Lambda bundles and assets
	DEPLOYMENT_BUCKET_PREFIX: ""
		Description: For publishing to a prefix in your deployment bucket, instead of root. Should
		include trailing slash e.g. 'my-prefix/'
	TARGET_REPO: "https://github.com/aws-samples/amazon-textract-transformer-pipeline"
		Target repository where your workshop code lives

	Common usage:

	...::: Build all SAM based services :::...
	$ make package

	...::: Deploy or re-deploy all SAM based services :::...
	$ make deploy

	...::: Create (cannot re-deploy) all SAM based services with rollback disabled :::...
	$ make create

	...::: Delete all SAM based services :::...
	$ make delete
endef
