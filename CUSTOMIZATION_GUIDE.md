# Applying and Customizing the Amazon Textract Transformer Pipeline

This file contains suggestions and considerations to help you apply and customize the sample to your own use cases.

> ⚠️ **Remember:** This repository is an illustrative sample, not intended as fully production-ready code. The guidance here is **not** an exhaustive path-to-production checklist.


## Bring your own dataset: Getting started step-by-step

So you've cloned this repository and reviewed the "Getting started" installation steps on [the README](README.md) - How can you get started with your own dataset instead of the credit card agreements example?

### Step 1: Any up-front CDK customizations

Depending on your use case you might want to make some customizations to the pipeline infrastructure itself. You can always revisit this later by running `cdk deploy` again to update your stack - but if you know up-front that some adjustments will be needed, you might choose to make them first.

Particular examples might include:

- Tuning to support **large documents**, especially if you'll be processing documents more than ~100-150 pages
- Enabling **additional online Textract features** (e.g. `TABLES` and/or `FORMS`) if you'll need them in online processing

For details on these and other use cases, see the **Customizing the pipeline** section below.

### Step 2: Deploy the stack and set up SageMaker

Follow the "Getting started" steps as outlined in the [the README](README.md) to deploy your pipeline and set up your SageMaker notebook environment with the sample code and notebooks - but don't start running through the notebooks just yet.

### Step 3: Clear out the sample annotations

In SageMaker, delete the provided `notebooks/data/annotations/augmentation-*` folders of pre-baked annotations on the credit card documents.

> **Why?:** The logic in notebook 1 for selecting a sample of documents to Textract and annotate automatically looks at your existing `data/annotations` to choose target files - so you'll see missing document errors if you don't delete these annotation files first.

### Step 4: Load your documents to SageMaker

Start running through [notebook 1](notebooks/1.%20Data%20Preparation.ipynb) but follow the guidance in the *"Fetch the data"* section to load your raw documents into the `notebooks/data/raw` folder in SageMaker **instead** of the sample CFPB documents.

How you load your documents into SageMaker may differ depending on where they're stored today. For example:

- If they're currently on your local computer, you should be able to drag and drop them to the folder pane in SageMaker/JupyterLab to upload.
- If they're currently on Amazon S3, you can copy a folder by running e.g. `!aws s3 sync s3://{DOC-EXAMPLE-BUCKET}/my/folder data/raw` from a cell in notebook 1.
- If they're currently compressed in a zip file, you can refer to the code used on the example data to help extract and tidy up the files.

> **Note:** Your customized notebook should still set the `rel_filepaths` variable which is used by later steps.

### Step 5: Customize the field configurations

In the *Defining the challenge* section of notebook 1, customize the `fields` list of `FieldConfiguration` definitions for the entities/fields you'd like to extract.

Each `FieldConfiguration` defines not just the name of the the field, but also other attributes like:

- The `annotation_guidance` that should be shown in the labelling tool to help workers annotate consistently
- How a single representative value should be `select`ed from multiple detected entity mentions, or else (implicit) that multiple values should be preserved
- Whether the field is mandatory on a document (implicit), or else `optional` or even completely `ignore`d

Your model will perform best if:

- It's possible to highlight your configured entities with complete consistency (no differences of opinion between annotators or tricky edge cases)
- Matches of your entities can be confirmed with local context somewhat nearby on the page (e.g. not dictated by a sentence right at the other end of the page, or on a previous page)

Configuring a large number of `fields` may increase required memory footprint of the model and downstream components in the pipeline, and could impact the accuracy of trained models. The SageMaker Ground Truth bounding box annotation tool used in the sample supports up to 50 labels as documented [in the SageMaker Developer Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/sms-bounding-box.html) - so attempting to configure more than 50 `fields` would require additional workarounds.

### Step 6: Enabling batch Textract features (if you want them)

In the *Textract the input documents* section of notebook 1, you'll see by default `features=[]` to optimize costs - since the SageMaker model and sample post-processing Lambda do not use or depend on additional Amazon Textract features like `TABLES` and `FORMS`.

If you'd like to enable these extra analyses for your batch processing in the notebook, set e.g. `features=["FORMS", "TABLES"]`. This setting is for the batch analysis only and does not affect the online behaviour of the deployed pipeline.

### Step 7: Customize how pages are selected for annotation

In the credit card agreements example, there's lots of data and no strong requirements on what to annotate. The code in the *Collect a dataset* section of notebook 1 selects pages from the corpus at random, but with a fixed (higher) proportion of first-page samples because the example entities are mostly likely to occur at the start of the document.

For your own use cases this emphasis on first pages may not apply. Moreover if you're strongly data- or time-constrained you might prefer to pick out a specific list of most-relevant pages to annotate!

Consider editing the `select_examples()` function to customize how the set of candidate document pages is chosen for the next annotation job, excluding the already-labelled `exclude_img_paths`.

### Step 8: Proceed with data annotation and subsequent steps

From the labelling job onwards (through notebook 2 and beyond), the flow should be essentially the same as with the sample data. Just remember to edit the `include_jobs` list in notebook 2 to reflect the actual annotation jobs you performed.

If your dataset is particularly tiny (more like e.g. 30 labelled pages than 100), it might be helpful to try increasing the `early_stopping_patience` hyperparameter to force the training job to re-process the same examples for longer. You could also explore hyperparameter tuning. However, it'd likely have a bigger impact to spend that time annotatting more data instead!


## Customizing the pipeline

### Handling large documents

Because some components of the pipeline have configured timeouts or process consolidated document Textract JSON in memory, scaling to very large documents (e.g. hundreds of pages) may require some configuration changes in the CDK solution.

Consider:

- Increasing the `timeout_excluding_queue` (in [pipeline/ocr/__init__.py TextractOcrStep](pipeline/ocr/__init__.py)) to accommodate the longer Textract processing and Lambda consolidation time (e.g. to 20mins+)
- Increasing the `timeout` and `memory_size` of the `CallTextract` Lambda function in [pipeline/ocr/__init__.py](pipeline/ocr/__init__.py) to accommodate consolidating the large Textract result JSON to a single S3 file (e.g. to 300sec, 1024MB)
- Likewise increasing the `memory_size` of the `PostProcessFn` Lambda function in [pipeline/postprocessing/__init__.py](pipeline/postprocessing/__init__.py), which also loads and processes full document JSON (e.g. to 1024MB)


### Using Amazon Textract `TABLES` and `FORMS` features in the pipeline

The sample SageMaker model and post-processing Lambda function neither depend on nor use the additional [tables](https://docs.aws.amazon.com/textract/latest/dg/how-it-works-tables.html) and [forms](https://docs.aws.amazon.com/textract/latest/dg/how-it-works-kvp.html) features of Amazon Textract and therefore by default they're disabled in the pipeline.

To enable these features for documents processed by the pipeline, you could for example:

- Add a key specifying `sfn_input["Textract"]["Features"] = ["FORMS", "TABLES"]` to the S3 trigger Lambda function in [pipeline/fn-trigger/main.py](pipeline/fn-trigger/main.py) to explicitly set this combination for all pipeline executions triggered by S3 uploads, OR
- Add a `DEFAULT_TEXTRACT_FEATURES=FORMS,TABLES` environment variable to the `CallTextract` Lambda function in [pipeline/ocr/__init__.py](pipeline/ocr/__init__.py) to make that the default setting whenever a pipeline run doesn't explicitly configure it.

Once the features are enabled for your pipeline, you can edit the post-processing Lambda function (in [pipeline/postprocessing/fn-postprocess](pipeline/postprocessing/fn-postprocess)) to combine them with your SageMaker model results as needed.

For example you could loop through the rows and cells of detected `TABLE`s in your document, using the SageMaker entity model results for each `WORD` to find the specific records and columns that you're interested in.

If you need to change the output format for the post-processing Lambda function, note that the A2I human review template will likely also need to be updated.


## Customizing the models

### How much data do I need?

As demonstrated in the credit card agreements example, the answer to this question is strongly affected by how "hard" your particular extraction problem is for the model. Particular factors that can make learning difficult include:

- Relying on important context "hundreds of words away" or across page boundaries - since each model inference analyzes a limited-length subset of the words on an individual page.
- Noisy or inconsistent annotations - disagreement or inconsistency between annotators is more common than you might initially expect, and this noise makes it difficult for the model to identify the right common patterns to learn.
- High importance of numbers or other unusual tokens - language models like these aren't well suited to mathematical analysis, so sense checks like "these line items should all add up to the total" or "this number should be bigger than that one" that may make sense to humans will not be obvious to the model.

A practical solution is to start small, and **run experiments with varying hold-out/validation set sizes**: Analyzing how performance changed with dataset size up to what you have today, can give an indication of how what incremental value you might get by continuing to collect and annotate more.

Depending on your task and document diversity, you may be able to start getting an initial idea of what seems "easy" with as few as 20 annotated pages, but may need hundreds to get a more confident view of what's "hard".


### Should I pre-train to my domain, or just fine-tune?

Because much more un-labelled domain data (Textracted documents) is usually available than labelled data (pages with manual annotations) for a use case, the compute infrastructure and/or time required for pre-training is usually significantly larger than fine-tuning jobs.

For this reason, it's typically best to **first experiment with fine-tuning the public base models** - before exploring what improvements might be achievable with domain-specific pre-training. It's also useful to understand the relative value of the effort of collecting more labelled data (see *How much data do I need?*) to compare against dedicating resources to pre-training.

Unless your domain diverges strongly from the data the public model was trained on (for example trying to transfer to a new language or an area heavy with unusual grammar/jargon), accuracy improvements from continuation pre-training are likely to be small and incremental rather than revolutionary. To consider from-scratch pre-training (without a public model base), you'll typically need a particularly large and diverse corpus (as per training BERT from scratch) to get good results.

> ⚠️ **Note:** Some extra code modifications may be required to pre-train from scratch (for example to initialize a new tokenizer from the source data).

There may be other factors aside from accuracy that influence your decision to pre-train. Notably:

- **Bias** - Large language models have been shown to learn not just grammar and semantics from datasets, but other patterns too: Meaning practitioners must consider the potential biases in source datasets, and what real-world harms this could cause if the model brings learned stereotypes to the target use case. For example see [*"StereoSet: Measuring stereotypical bias in pretrained language models", (2020)*](https://arxiv.org/abs/2004.09456)
- **Privacy** - In some specific circumstances (for example [*"Extracting Training Data from Large Language Models", 2020*](https://arxiv.org/abs/2012.07805)) it has been shown that input data can be reverse-engineered from trained large language models. Protections against this may be required, depending how your model is exposed to users.
