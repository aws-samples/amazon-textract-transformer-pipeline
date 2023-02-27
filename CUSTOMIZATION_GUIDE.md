# Applying and Customizing the Amazon Textract Transformer Pipeline

This file contains suggestions and considerations to help you apply and customize the sample to your own use cases.

> ⚠️ **Remember:** This repository is an illustrative sample, not intended as fully production-ready code. The guidance here is **not** an exhaustive path-to-production checklist.

## Contents

1. [Bring your own dataset guidance](#byo-data)
1. [Customizing the pipeline](#customizing-pipeline)
    - [Skipping page image generation (optimizing for LayoutLMv1)](#skip-thumbnails)
    - [Auto-scaling SageMaker endpoints](#auto-scaling)
    - [Handling large documents (or optimizing for small ones)](#handling-large-docs)
    - [Using Amazon Textract `TABLES` and `FORMS` features in the pipeline](#using-analyzedocument-features)
    - [Using alternative OCR engines](#altern-ocr)
1. [Customizing the models](#customizing-models)
    - [How much data do I need?](#how-much-data)
    - [Should I pre-train to my domain, or just fine-tune?](#should-i-pre-train)
    - [Scaling and optimizing model training](#optimizing-model-training)

---

## Bring your own dataset: Getting started step-by-step<a id='byo-data'></a>

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

In the *OCR the input documents* section of notebook 1, you'll see by default `features=[]` to optimize costs - since the SageMaker model and sample post-processing Lambda do not use or depend on additional Amazon Textract features like `TABLES` and `FORMS`.

If you'd like to enable these extra analyses for your batch processing in the notebook, set e.g. `features=["FORMS", "TABLES"]`. This setting is for the batch analysis only and does not affect the online behaviour of the deployed pipeline.

### Step 7: Customize how pages are selected for annotation

In the credit card agreements example, there's lots of data and no strong requirements on what to annotate. The code in the *Collect a dataset* section of notebook 1 selects pages from the corpus at random, but with a fixed (higher) proportion of first-page samples because the example entities are mostly likely to occur at the start of the document.

For your own use cases this emphasis on first pages may not apply. Moreover if you're strongly data- or time-constrained you might prefer to pick out a specific list of most-relevant pages to annotate!

Consider editing the `select_examples()` function to customize how the set of candidate document pages is chosen for the next annotation job, excluding the already-labelled `exclude_img_paths`.

### Step 8: Proceed with data annotation and subsequent steps

If you're planning to review OCR accuracy as part of your PoC or train models to normalize from the raw detected text to standardised values (for example, normalising dates or number representations), you might find it useful to use the custom Ground Truth UI presented in Notebook 1 - instead of the default built-in (bounding-box only) UI.

From the labelling job onwards (through notebook 2 and beyond), the flow should be essentially the same as with the sample data. Just remember to edit the `include_jobs` list in notebook 2 to reflect the actual annotation jobs you performed.

If your dataset is particularly tiny (more like e.g. 30 labelled pages than 100), it might be helpful to try increasing the `early_stopping_patience` hyperparameter to force the training job to re-process the same examples for longer. You could also explore hyperparameter tuning. However, it'd likely have a bigger impact to spend that time annotatting more data instead!

---

## Customizing the pipeline<a id='customizing-pipeline'></a>

### Skipping page image generation (optimizing for LayoutLMv1)<a id='skip-thumbnails'></a>

By default, the deployed pipeline will invoke a page thumbnail image generation endpoint in parallel to running input documents through Amazon Textract. These are useful additional inputs for some model architectures supported by the pipeline (e.g. LayoutLMv2, LayoutXLM), but are not required or used by LayoutLMv1.

If you know you'll be using the pipeline with LayoutLMv1 models only, you can `export USE_THUMBNAILS=false` before deploying your app (or edit [cdk_app.py](cdk_app.py) to set `use_thumbnails=False`) to remove the parallel thumbnailing step from the pipeline. When the pipeline is deployed with the default `use_thumbnails=True`, it will fail unless a thumbnailing endpoint is properly configured (via SSM as shown in [notebook 2](notebooks/2.%20Model%20Training.ipynb)).


### Auto-scaling SageMaker endpoints<a id='auto-scaling'></a>

If your pipeline will see variable load - *especially* if there will be long periods where no documents are submitted at all - then you might be interested to optimise resource use and cost by enabling infrastructure auto-scaling on deployed SageMaker endpoints.

Scaling down to 0 instances during idle periods is [supported](https://docs.aws.amazon.com/sagemaker/latest/dg/async-inference-autoscale.html) by the asynchronous endpoints we use in this example - but there's a trade-off: You may see a cold-start delay of several minutes when a document arrives and no instances were already active.

For endpoints automatically deployed by the CDK app, you can control whether auto-scaling is set up via the `ENABLE_SM_AUTOSCALING` environment variable or the `enable_sagemaker_autoscaling` argument in [cdk_app.py](cdk_app.py). For instructions to set up auto-scaling on your manually-created endpoints, see [notebooks/Optional Extras.ipynb](notebooks/Optional%20Extras.ipynb).


### Handling large documents (or optimizing for small ones)<a id='handling-large-docs'></a>

Because some components of the pipeline have configured timeouts or process consolidated document Textract JSON in memory, scaling to very large documents (e.g. hundreds of pages) may require some configuration changes in the CDK solution.

Consider:

- Increasing the default `timeout_excluding_queue` (in [pipeline/ocr/textract_ocr.py TextractOCRStep](pipeline/ocr/textract_ocr.py)) to accommodate the longer Textract processing and Lambda consolidation time (e.g. to 20mins+)
- Increasing the `timeout` and `memory_size` of the `CallTextract` Lambda function in [pipeline/ocr/__init__.py](pipeline/ocr/__init__.py) to accommodate consolidating the large Textract result JSON to a single S3 file (e.g. to 8min, 2048MB)
- Likewise increasing the `memory_size` of the `PostProcessFn` Lambda function in [pipeline/postprocessing/__init__.py](pipeline/postprocessing/__init__.py), which also loads and processes full document JSON (e.g. to 1024MB or more)

Conversely if you know up-front your use case will handle only images or short documents, you may be able to reduce these settings from the defaults to save costs.


### Using Amazon Textract `TABLES` and `FORMS` features in the pipeline<a id='using-analyzedocument-features'></a>

The sample SageMaker model and post-processing Lambda function neither depend on nor use the additional [tables](https://docs.aws.amazon.com/textract/latest/dg/how-it-works-tables.html) and [forms](https://docs.aws.amazon.com/textract/latest/dg/how-it-works-kvp.html) features of Amazon Textract and therefore by default they're disabled in the pipeline.

To enable these features for documents processed by the pipeline, you could for example:

- Add a key specifying `sfn_input["Textract"]["Features"] = ["FORMS", "TABLES"]` to the S3 trigger Lambda function in [pipeline/fn-trigger/main.py](pipeline/fn-trigger/main.py) to explicitly set this combination for all pipeline executions triggered by S3 uploads, OR
- Add a `DEFAULT_TEXTRACT_FEATURES=FORMS,TABLES` environment variable to the `CallTextract` Lambda function in [pipeline/ocr/__init__.py](pipeline/ocr/__init__.py) to make that the default setting whenever a pipeline run doesn't explicitly configure it.

Once the features are enabled for your pipeline, you can edit the post-processing Lambda function (in [pipeline/postprocessing/fn-postprocess](pipeline/postprocessing/fn-postprocess)) to combine them with your SageMaker model results as needed.

For example you could loop through the rows and cells of detected `TABLE`s in your document, using the SageMaker entity model results for each `WORD` to find the specific records and columns that you're interested in.

If you need to change the output format for the post-processing Lambda function, note that the A2I human review template will likely also need to be updated.


### Using alternative OCR engines<a id='altern-ocr'></a>

For the highest accuracy, broadest feature set, and least maintenance overhead on supported languages - you'll typically want to use the [Amazon Textract service](https://aws.amazon.com/textract/) for raw document text extraction, which is the default for this solution.

Some use-cases may want to explore other options though: Particularly if you're dealing with **low-resource-languages** not currently supported by Textract (see *"What type of text can Amazon Textract detect and extract?"* in the [service FAQ](https://aws.amazon.com/textract/faqs/)). In general, if your documents use an unsupported language but a *supported character* set (for example **Indonesian**), you'll be better off using Amazon Textract... But if your documents rely heavily on **unsupported characters** (for example **Thai**), an alternative will be needed.

This solution includes integration points for deploying alternative, open-source-based OCR engines on Amazon SageMaker endpoints: And an example integration with [Tesseract OCR](https://github.com/tesseract-ocr/tesseract).

**To deploy and use the Tesseract OCR engine:**

1. Configure the default languages for Tesseract:
    - Edit `CUSTOM_OCR_ENGINES` in [pipeline/ocr/sagemaker_ocr.py](pipeline/ocr/sagemaker_ocr.py) to set `OCR_DEFAULT_LANGUAGES` to the comma-separated [Tesseract language codes](https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html) needed for your use-case.
    - ⚠️ This languages parameter is not exposed through the CloudFormation bootstrap stack. If you're using that to run your CDK deployment, you'll need to fork and edit the repository.
1. Before (re)-running `cdk deploy`:
    - Configure your CDK stack to **build** a Tesseract container & SageMaker Model: `export BUILD_SM_OCRS=tesseract`, or refer to the `build_sagemaker_ocrs` parameter in [cdk_app.py](cdk_app.py)
    - Configure your CDK stack to **deploy** a SageMaker endpoint for Tesseract: `export DEPLOY_SM_OCRS=tesseract`, or refer to the `deploy_sagemaker_ocrs` parameter in cdk_app.py.
    - Configure your CDK pipeline to **use** a SageMaker endpoint instead of Amazon Textract: `export USE_SM_OCR=tesseract`, or refer to the `use_sagemaker_ocr` parameter in cdk_app.py.
    - If you're using the CloudFormation bootstrap stack for this example, you can configure these parameters via CloudFormation.

**To integrate other custom/alternative OCR engines via SageMaker:**

- Add your own entries to `CUSTOM_OCR_ENGINES` in [pipeline/ocr/sagemaker_ocr.py](pipeline/ocr/sagemaker_ocr.py), which defines the container image, script bundle location, environment variables, and other configurations - to be used for each possible engine.
    - You can refer to the existing `tesseract` definition for an example, and choose your own name for your custom engine.
    - You may like to re-use and extend the existing [Dockerfile](notebooks/custom-containers/preproc/Dockerfile), which is already parameterized to optionally install Tesseract via a build arg.
- Develop your integration scripts in [notebooks/preproc/textract_transformers/ocr_engines](notebooks/preproc/textract_transformers/ocr_engines):
    - Create a new `eng_***.py` script to define your engine, using the existing [eng_tesseract.py](notebooks/preproc/textract_transformers/ocr_engines/eng_tesseract.py) as a guide.
    - Your script should define one class based on `base.BaseOCREngine`, implementing the `process()` method with the expected arguments and return type.
    - Your `process()` method should return an Amazon Textract-compatible (JSONable) dictionary. Use the `generate_response_json()` utility and other classes from [base.py](notebooks/preproc/textract_transformers/ocr_engines/base.py) to help with this.
- Once your engine is ready, you can deploy it into your stack with `BUILD_SM_OCRS`, `DEPLOY_SM_OCRS` and `USE_SM_OCR` as described above.
    - Note that the build and deploy options support multiple engines (comma-separated string), which you can use to test out alternative model and endpoint deployments. However, the stack can only use one OCR engine (or Amazon Textract) at a time.
- 🚀 Feel free to pull-request your alternative integrations!


---

## Customizing the models<a id='customizing-models'></a>

### How much data do I need?<a id='how-much-data'></a>

As demonstrated in the credit card agreements example, the answer to this question is strongly affected by how "hard" your particular extraction problem is for the model. Particular factors that can make learning difficult include:

- Relying on important context "hundreds of words away" or across page boundaries - since each model inference analyzes a limited-length subset of the words on an individual page.
- Noisy or inconsistent annotations - disagreement or inconsistency between annotators is more common than you might initially expect, and this noise makes it difficult for the model to identify the right common patterns to learn.
- High importance of numbers or other unusual tokens - language models like these aren't well suited to mathematical analysis, so sense checks like "these line items should all add up to the total" or "this number should be bigger than that one" that may make sense to humans will not be obvious to the model.

A practical solution is to start small, and **run experiments with varying hold-out/validation set sizes**: Analyzing how performance changed with dataset size up to what you have today, can give an indication of how what incremental value you might get by continuing to collect and annotate more.

Depending on your task and document diversity, you may be able to start getting an initial idea of what seems "easy" with as few as 20 annotated pages, but may need hundreds to get a more confident view of what's "hard".


### Should I pre-train to my domain, or just fine-tune?<a id='should-i-pre-train'></a>

Because much more un-labelled domain data (Textracted documents) is usually available than labelled data (pages with manual annotations) for a use case, the compute infrastructure and/or time required for pre-training is usually significantly larger than fine-tuning jobs.

For this reason, it's typically best to **first experiment with fine-tuning the public base models** - before exploring what improvements might be achievable with domain-specific pre-training. It's also useful to understand the relative value of the effort of collecting more labelled data (see *How much data do I need?*) to compare against dedicating resources to pre-training.

Unless your domain diverges strongly from the data the public model was trained on (for example trying to transfer to a new language or an area heavy with unusual grammar/jargon), accuracy improvements from continuation pre-training are likely to be small and incremental rather than revolutionary. To consider from-scratch pre-training (without a public model base), you'll typically need a large and diverse corpus (as per training BERT from scratch) to get good results.

> ⚠️ **Note:** Some extra code modifications may be required to pre-train from scratch (for example to initialize a new tokenizer from the source data).

There may be other factors aside from accuracy that influence your decision to pre-train. Notably:

- **Bias** - Large language models have been shown to learn not just grammar and semantics from datasets, but other patterns too: Meaning practitioners must consider the potential biases in source datasets, and what real-world harms this could cause if the model brings learned stereotypes to the target use case. For example see [*"StereoSet: Measuring stereotypical bias in pretrained language models", (2020)*](https://arxiv.org/abs/2004.09456)
- **Privacy** - In some specific circumstances (for example [*"Extracting Training Data from Large Language Models", 2020*](https://arxiv.org/abs/2012.07805)) it has been shown that input data can be reverse-engineered from trained large language models. Protections against this may be required, depending how your model is exposed to users.


### Scaling and optimizing model training<a id='optimizing-model-training'></a>

While fine-tuning is often practical on a single-GPU instance like `ml.g4dn.xlarge` or `ml.p3.2xlarge`, you'll probably be interested in scaling up for the larger datasets involved in pre-training. Here are some tips to be aware of:

1. The default **multi-GPU** behavior of Hugging Face (v4.17) `Trainer` on SageMaker is normally [DataParallel (DP)](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html) in standard "script mode", but (AWS' optimized implementation of) [DistributedDataParallel (DDP)](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) will be launched when using [SageMaker Distributed Data Parallel](https://docs.aws.amazon.com/sagemaker/latest/dg/data-parallel.html) on a supported instance type (such as `ml.p3.16xlarge` or larger)
    - The underlying implementation of LayoutLMv2 (and by extension, LayoutXLM which is based on the same architecture), [does not support](https://github.com/huggingface/transformers/issues/14110#issuecomment-949419662) DP mode. Therefore to train LayoutXLM/v2 on a multi-GPU instance you must **either**:
        - Use SageMaker DDP by selecting a supported instance type (e.g. `ml.p3.16xlarge`) and setting `distribution={ "smdistributed": { "dataparallel": { "enabled": True } } },`, **OR**
        - Use the `ddp_launcher.py` script as entrypoint instead of `train.py`, to launch native PyTorch DDP.
      to enable SageMaker DDP, should.
    - ⚠️ **Warning:** At some points in development, a bug was observed where multi-worker data pre-processing would hang with SageMaker DDP. One workaround is to set `dataproc_num_workers` to `1`, but if your dataset is large it will probably take longer to load in single-worker than the 30 minute default AllReduce time-out. This issue *may* still be present, so please raise it if you come across it.
    - LayoutLMv1 can train in either mode, but DP adds additional GPU memory overhead to the "lead" GPU, so you may be able to run larger batch sizes before encountering `CUDAOutOfMemory` if using DDP: Either by using the native PyTorch `ddp_launcher.py` or SageMaker `smdistributed`.
1. Remember that `per_device_train_batch_size` controls **per-accelerator batching**
    - Moving up to a multi-GPU instance type implicitly increases the overall batch size for learning
    - Smaller models like LayoutLMv1 may be able to tolerate larger per-device batch sizes than LayoutLMv2/XLM before running out of GPU memory: For example train/eval settings of `4/8` may be appropriate for LLMv1 as opposed to `2/4` for v2/XLM.
1. On job start-up, input manifests and Textract JSONs are **pre-processed** into a training-ready dataset. This is managed by the "lead" process with `dataproc_num_workers`, and the processes mapping to other GPUs wait until it's complete to load the data from cache files. Consequences of this include:
    - `dataproc_num_workers` can be configured to consume (almost) as many CPU cores as the machine has available, regardless of how many GPU devices are present in DDP mode, and will be highly parallel by default.
    - If data processing doesn't complete within [the default DDP timeout](https://discuss.pytorch.org/t/how-to-set-longer-timeout-for-ddp-training/143213) (usually 30min), the other GPU manager processes will stop waiting and the job will fail. This should only be a problem if your dataset is extremely large, or you're trying to train with a very restricted `dataproc_num_workers` (or instance CPU count) as compared to the dataset size.
    - In principle you could consider refactoring to use a separate SageMaker Processing job for this initial setup, and save the resultant `datasets` cache files to S3 to use as training job inputs. A [SageMaker Pipeline](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-sdk.html) could be used to orchestrate these two jobs together and even cache processing results between training runs. We chose not to do this initially, to keep the end-to-end flow simpler and the training job inputs more intuitive.
1. Multiprocessing (especially combining multiple types of multiprocessing) can sometimes cause **deadlocks and swallowed error logs**
    - Setting a sensible [`max_run` timeout](https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html#sagemaker.estimator.EstimatorBase) can help minimize wasted resources even if a deadlock prevents your job from properly stopping after an unhandled error.
    - Perform initial testing on a smaller subset of target data, perhaps disabling some parallelism controls (like setting hyperparameters `"dataproc_num_workers": 0,` and `"dataloader_num_workers": 0,`) at first to better surface the cause of any error messages. Checking functionality on a single-GPU training run before scaling to DDP training may also help.
    - Other things that may help surface diagnostic messages include setting environment variable `PYTHONUNBUFFERED=1` and substituting `logger` calls for `print()` - to help ensure messages get published before a crashing thread exits.
1. If your document filepaths *don't contain special characters*, then you may be able to improve job start-up performance using **[Fast File Mode](https://aws.amazon.com/about-aws/whats-new/2021/10/amazon-sagemaker-fast-file-mode/)** by setting `input_mode="FastFile"` on the Estimator (or configuring per-channel in `fit()`). The Credit Card Agreements example document filenames *do* though, and this has been observed to cause errors with FastFile mode in testing - so is not turned on below by default.
1. If you're interested to explore **[SageMaker Training Compiler](https://docs.aws.amazon.com/sagemaker/latest/dg/training-compiler.html)** to further optimize training of these models:
    - Check out the additional guidance in [src/smtc_launcher.py](notebooks/src/smtc_launcher.py)
    - ...But be aware in our initial tests we found errors getting the compiler to work with LayoutLMv2+, and although it functionally worked with LayoutLMv1 a re-compilation was being triggered each epoch - which offset performance gains since epochs themselves were fairly short.
