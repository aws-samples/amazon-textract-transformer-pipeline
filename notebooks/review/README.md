# Custom Document Review UI for Amazon A2I with VueJS

> _"Wow, this folder looks complicated - what's going on?" - Anonymous Developer_

Don't panic! You're free to just consume the final template as-is for the sample demo. However:

- This example is structured to show **more general patterns** and exploration of implementing complex custom A2I task UIs - rather than providing a minimal or optimal implementation of the given functionality.
- This readme discusses the context and design considerations, to help you **scalably and deeply customize the sample** for your own needs.


## Quick start how-to

Assuming you have [Node.js installed](https://nodejs.org/en/download/), then from this folder you can run:

- `npm install` to install the required dependencies
- `npm run dev` to start continuously monitoring, building and serving the app as you make changes (see http://localhost:3000 and http://localhost:3000/index-noliquid.html in your browser if running this locally, or refer to the "Building in SageMaker" section below in SageMaker)
- `npm run build` to build the final A2I-ready template file to `dist/index.html`

Once the final template is built, you can test it out further with the [SageMaker RenderUiTemplate API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_RenderUiTemplate.html) (via AWS SDKs for [Python](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker.html#SageMaker.Client.render_ui_template), [JavaScript](https://docs.aws.amazon.com/AWSJavaScriptSDK/v3/latest/clients/client-sagemaker/classes/renderuitemplatecommand.html), or others) before loading in to Amazon A2I or Amazon SageMaker Ground Truth.

By default, `index-noliquid.html` is set up to expect an `example.pdf` in the `public/` folder but this is not provided in the repository. You can bring your own PDF to test it out.


## The need for front-end frameworks

At the time of writing, the majority of samples you'll see over on the [aws-samples/amazon-a2i-sample-task-uis](https://github.com/aws-samples/amazon-a2i-sample-task-uis) and [aws-samples/amazon-sagemaker-ground-truth-task-uis](https://github.com/aws-samples/amazon-sagemaker-ground-truth-task-uis) repositories (as well as most discussion in the SageMaker Developer Guide - such as [here](https://docs.aws.amazon.com/sagemaker/latest/dg/sms-custom-templates-step2.html)) deal with **simple templates created directly in HTML**.

This helps with getting started quickly and focussing on the specifics of SageMaker integrations, but it's probably fair to say that most web UI development today is not done in this direct way but with some **supporting toolchain**. Benefits of abstracting development further away from the final browser-ready document can include:

- Automatic transpilation and compilation, allowing you to write code using newer syntax features like [TypeScript](https://www.typescriptlang.org/) and [ECMAScript improvements](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Language_Resources) **independently** of which older browser versions you aim to support.
- Integration with **package management and bundling tools** to help you consume and manage external dependencies, and keep your own code modular and maintainable.
- Using **frameworks** to simplify your code and align towards accepted best practices (for example around state and reactivity + event management): Helping keep more complex projects manageable. Some popular, component-based web UI frameworks include [React](https://reactjs.org/), [Vue.js](https://vuejs.org/) and [Angular](https://angular.io/).

A trade-off, of course, is the need to pick up a few extra front-end technologies and navigate integrating them with the SageMaker interfaces.

So how can you start developing more complex task UI templates? Luckily, this sample and documentation is here to give some guidance! This stack uses (amongst other tools):

- [Vue.js v3](https://vuejs.org/) web UI framework with [TypeScript](https://www.typescriptlang.org/)
- [Vite](https://vitejs.dev/) and [Rollup](https://rollupjs.org/guide/en/) for fast, configurable builds
- [Sass](https://sass-lang.com/guide) for feature-rich CSS style pre-processing


## Integration step-by-step: Vue.js components with SageMaker Ground Truth/A2I

For much of this document we'll talk about Amazon SageMaker Ground Truth (SMGT, for batch data labelling) and Amazon Augmented AI (A2I, for online human review) somewhat interchangeably because the core custom UI template integration patterns are largely the same between the two.

SMGT/A2I task templates are [**Liquid HTML** templates](https://shopify.github.io/liquid/), which is to say that for the overall flow:

1. SageMaker receives the input object (something JSON-like, whose exact structure will depend on the task definition) for the particular task the user is picking up
2. SageMaker resolves your liquid template to render the final HTML - populating placeholders with data for the specific task as necessary
3. The final HTML (presumably including the special `<crowd-form>` component and perhaps other [Crowd HTML Elements](https://docs.aws.amazon.com/sagemaker/latest/dg/sms-ui-template-reference.html)) is loaded in the worker's browser within the SMGT/A2I portal
4. The worker does what they need to do and then submits the form.
5. The form data is collected and post-processed into the output of the task (SMGT output manifest or A2I task result JSON). The worker moves on to the next task or object/document/image/etc, and the cycle repeats.


### Step 1: The task input object

The task input object structure will be defined by your input JSON-Lines manifest file and [pre-processing Lambda function](https://docs.aws.amazon.com/sagemaker/latest/dg/sms-custom-templates-step3.html) for SageMaker Ground Truth labeling jobs; or the [input data content](https://docs.aws.amazon.com/sagemaker/latest/dg/a2i-start-human-loop.html#a2i-instructions-starthumanloop) for an Amazon A2I human loop.

In both cases ([SMGT Doc](https://docs.aws.amazon.com/sagemaker/latest/dg/sms-custom-templates-step2.html#sms-custom-templates-step2-automate), [A2I Doc](https://docs.aws.amazon.com/sagemaker/latest/dg/a2i-custom-templates.html#a2i-custom-template-step2-UI-vars)), your core input data will be under the `task.input` object in Liquid context.


### Step 2: Liquid template fulfillment

Your template will be processed by SageMaker to substitute values based on the input task object. For example given a task object like:

```json
{
  "fields": [
    { "Name": "Field1", "Value": "Hello" },
    { "Name": "Field2", "Value": "World" },
  ]
}
```

And a template in [Liquid](https://shopify.github.io/liquid/) like this (note the `{{` and `{%` syntax elements, and the fact that these are valid in the whole document including script tags):

```html
<script>
  var myCoolVariable = "{{ task }}";
  console.log(  // Unescape and log full task contents
    JSON.parse(
      DOMParser()
        .parseFromString(myCoolVariable, "text/html")
        .documentElement
        .textContent
    )
  );
</script>
<crowd-form>
  <h1>I am a task template</h1>
  {% for field in task.input.fields %}
  <my-field class="example" name="{{field.Name}}">
    {{field.Value}}
  </my-field>
  {% endfor %}
</crowd-form>
```

The final HTML would be something like:

```html
<script>
  var myCoolVariable = "{&quot;input&quot;:{&quot;fields&quot;:..." [...]
  console.log(  // Unescape and log full task contents
    JSON.parse(
      DOMParser()
        .parseFromString(myCoolVariable, "text/html")
        .documentElement
        .textContent
    )
  );
</script>
<crowd-form>
  <h1>I am a task template</h1>
  <my-field class="example" name="Field1">
    Hello
  </my-field>
  <my-field class="example" name="Field2">
    World
  </my-field>
</crowd-form>
```

This is how you'll pass task data through to your UI at run-time, and it has several important consequences:

▶️ **Tip 1: Use the SageMaker rendering API for fast Liquid development and debugging**

You **don't need** to create a labeling job or human loop to test your Liquid template out, and it would likely be a slow way to work. The [SageMaker RenderUiTemplate API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_RenderUiTemplate.html) (via AWS SDKs for [Python](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker.html#SageMaker.Client.render_ui_template), [JavaScript](https://docs.aws.amazon.com/AWSJavaScriptSDK/v3/latest/clients/client-sagemaker/classes/renderuitemplatecommand.html), or others) can do it for you quickly and easily.

▶️ **Tip 2: Keep to valid-HTML features of Liquid**

You build your task UI as a Liquid HTML template, *then* it gets resolved at runtime. Most web UI build toolchains need to **parse the HTML** to some extent, which is fine for many Liquid features but not all.

The examples above were all okay because Liquid syntax appeared in places where it could be interpreted as either text in the document or HTML element attribute content... But consider the following:

```
<!-- Valid Liquid, Invalid HTML -->
<my-field {% if task.input.shouldDisable %} disabled {% endif %}>
  Oh no!
</my-field>
```

While this would be fine to include in your SageMaker task template, most bundling tools (including the Vite-based stack here) would throw an error on this unexpected content *within a tag itself*.

▶️ **Tip 3: Avoid unintended Liquid template notation from your frameworks**

Liquid uses `{{ }}` and `{% %}` to delimit its expressions, except in any sections of the file you [exclude](https://shopify.github.io/liquid/tags/template/) by wrapping with `{% raw %}` and `{% endraw %}` or `{% comment %}` and `{% endcomment %}`.

You'll need to consider this if working with UI frameworks which use these notations *in such a way that the characters appear in the output HTML file*. By default, Vue.js templates use "handlebars" `{{ }}` notation which is passed through to generated JavaScript and therefore, ***if*** script assets are inlined (see "single-file bundling" below), also appear in the bundled HTML.

In this sample we address the issue by configuring Vue to use `${ }` instead of `{{ }}` in templates: Setting the [app.compilerOptions.delimiters](https://vuejs.org/api/application.html#app-config-compileroptions) option in [vite.config.js](vite.config.js).

Teams with a strong preference for keeping the default Vue notation could instead:

1. Avoid inlining your built Vue JS scripts to the main HTML template file in the build process (instead hosting them elsewhere to be fetched directly from workers' browsers on page load), **or**
2. Explore options and customizations in your build toolchain for ways (for example, Vite plugins) to wrap inlined script assets with `{% raw %}` and `{% endraw %}` in the output.


▶️ **Tip 4: Consider using both an HTML test page and your main liquid template**

For developing front-end components, you'll want to be able to preview your changes as you make them - quickly, and ideally automatically. Unless you can integrate the SageMaker render API directly into your dev toolchain (not done in this example, but potentially possible), you might find it helpful to work with two "entry point" HTML files: One with Liquid expressions which will build your final UI template (like our [index.html](index.html)), and one plain HTML where the placeholders are replaced to exercise the range of functionality (like our [index-noliquid.html](index-noliquid.html)).


### Step 3: The task is rendered in the worker's browser

After your Liquid template is processed into final HTML, it's downloaded to and opened in the worker's browser, where your chosen front-end JS framework will run to deliver rich interactivity.

▶️ **Tip 5: You choose the balance between templating and UI framework logic**

Given that:

- Your UI will need to use *at least some* of SageMaker's [Crowd HTML Elements](https://docs.aws.amazon.com/sagemaker/latest/dg/sms-ui-template-reference.html) (for collecting the result in later steps), and
- assuming you're going to build some UI components that actually act on the task input data

...the UI HTML cannot be as simple as the vanilla `<div id="app"><!-- Mount app here --></div>` you might see in framework quickstart guides (as [here for Vue](https://vuejs.org/guide/introduction.html)).

However as we showed above, there are multiple ways you could use Liquid to inject task data into your web page: From controlling the overall structure of elements, to setting their attributes, to simply setting JavaScript variables in the page's context.

As a result, you have some flexibility (with trade-offs that we'll discuss further below) to choose how much of this dynamic layout you implement by the Liquid template (lots of conditional expressions to create or set parameters on elements), and how much you implement in your UI app (simple Liquid data passing, Vue-based client-side display logic).

▶️ **Tip 6: Integrating your web UI framework alongside Web Components**

Component-based web frameworks like Vue and React provide specific, opinionated tools to help you build web apps that can scale to complex use cases. [Web Components](https://developer.mozilla.org/en-US/docs/Web/Web_Components) refers to a suite of *browser native* technologies designed to support similar aims without ties to a particular framework. The [SageMaker Crowd HTML Elements](https://docs.aws.amazon.com/sagemaker/latest/dg/sms-ui-template-reference.html) (like `<crowd-form>`, `<crowd-checkbox>`, etc) are Web Components.

As mentioned in both the [Vue](https://vuejs.org/guide/extras/web-components.html) and [React](https://reactjs.org/docs/web-components.html) documentation, these UI frameworks are *complementary* to Web Components: **You can use Web Components in Vue/React, and use Vue/React to build Web Components**. However, the APIs and conventions for common tasks like passing events and processing updates will likely be **different** between native components and your chosen framework: For example [Vue events](https://vuejs.org/guide/essentials/event-handling.html) vs native [DOM CustomEvents](https://developer.mozilla.org/en-US/docs/Web/API/CustomEvent).

The balance you choose between implementing your UI in one big Vue/React/etc app context, or making heavy use of Liquid templating to set up more modular components - will influence how much your code looks like a "normal" Vue/React/etc app, versus having more emphasis on Web Component interactions.


### Step 4: Worker performs the task and submits the form

So far we've covered creating the UI and passing data **in**, but what about task **results**?

The [`<crowd-form>` element](https://docs.aws.amazon.com/sagemaker/latest/dg/sms-ui-template-crowd-form.html) is a wrapper for custom tasks that handles collection and submission of results to SageMaker, somewhat like a native HTML [`<form>` element](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/form).

The form should automatically detect and include both native input elements (like `<input>`, `<textarea>`, `<select>`) and crowd HTML input elements (like `<crowd-checkbox>`, `<crowd-slider>` and so on). However, there's one important limitation: **Queries and forms do not traverse DOM shadow roots**.

This means that if you're building components with a framework like Vue which use [shadow DOM](https://developer.mozilla.org/en-US/docs/Web/Web_Components/Using_shadow_DOM), the default behaviour will probably be that any inputs you nest *inside* your components will be invisible to the form and not picked up in the results!

▶️ **Tip 7: Exposing shadow-DOM inputs to your `<crowd-form>`**

In some frameworks it may be possible to disable the use of shadow DOM to work around this issue - but at the time of writing I couldn't find a way to do this in Vue.js v3.

There's also an interesting new proposed [ElementInternals interface](https://developer.mozilla.org/en-US/docs/Web/API/ElementInternals#specifications) in the [HTML Standard](https://html.spec.whatwg.org/multipage/custom-elements.html#the-elementinternals-interface) which would provide a mechanism for top-level custom components to report their inner data up to the form. A [polyfill](https://www.npmjs.com/package/element-internals-polyfill) for older browsers is even available on NPM. However when I tried this pattern for individual controls (left in 'FieldSingleValue' in this sample for demonstration), I found it worked as intended in Firefox but not Chrome.

Instead, the main workaround used in this sample is as follows:

- Create a custom component with the special tag name `<object-value-input>` (which is recognised by `<crowd-form>`), and put an instance of it under the form but outside any components which use shadow DOM.
- Use this as a proxy via some kind of **state store** pattern: So that any components buried in your UI which need to output result data bind to it and keep it updated to match their own state.
- Optionally, also process `validate()` events on the `<object-value-input>` so they propagate to all the custom input components. In this sample, this is used to check that all `required` fields have been filled in before submitting.

You can check what would be submitted by pressing the 'Submit' button while previewing the web page.


### Step 5: Result data post-processing

There's not much to say on this for A2I: Your task output will be delivered [to Amazon S3](https://docs.aws.amazon.com/sagemaker/latest/dg/a2i-create-flow-definition.html#a2i-create-human-review-api) with some additional metadata. In SageMaker Ground Truth, your [post-processing Lambda function](https://docs.aws.amazon.com/sagemaker/latest/dg/sms-custom-templates-step3-lambda-requirements.html#sms-custom-templates-step3-postlambda) might do some additional transformations (such as merging results from multiple workers) before saving the result to the output manifest file.


## Other considerations

### Dependency externalization and single-file bundling

With **bundling** tools like Vite and Rollup.js, the default behaviour is for any dependency libraries you install to your project (`npm install vue`) and then use in your app (`import {...} from vue`) to be bundled in to your built assets `dist/`. It's also common for these outputs assets to be split across several files (HTML, scripts, stylesheets, etc) which helps page load performance by allowing users' browsers to parallelize executing initial code with fetching other remaining assets.

This is great if you want full control over the hosting of all components needed by your UI, and have somewhere available to host extra assets (such as an [Amazon CloudFront distribution](https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/Introduction.html) or [S3 Bucket website](https://docs.aws.amazon.com/AmazonS3/latest/userguide/WebsiteHosting.html)) - but what if you'd rather avoid managing that extra infrastructure?

This sample uses the [vite-plugin-singlefile](https://www.npmjs.com/package/vite-plugin-singlefile) plugin to package all generated script and style assets into the main `index.html`, so that nothing needs self-hosting beyond uploading the UI template HTML.

To keep that file reasonably small, we also (manually) externalize dependencies using one of the several free Content Delivery Network (CDN) services that mirror NPM, [jsDelivr](https://www.jsdelivr.com/):

- External `<script>` tags are added to the source HTML to load the resources from CDN, with an integrity hash check to guard against possible tampering (these hashes be generated with a site like https://www.srihash.org/, if you have an URL that's trusted at current point in time).
- The import targets are declared as `external` in [vite.config.js](vite.config.js) and mapped to whatever global variable is created when loading the library direct in browser (refer to each library's documentation).
- Externalized dependencies are pinned to exact versions in NPM [package.json](package.json), to avoid any potential hard-to-debug errors caused by a mismatch.


### "Can vs Should": Gaps between Vue and Web Components

As mentioned above, you have some flexibility when integrating Vue into SMGT/A2I templates - to decide to what extent you'd like to:

1. Use Liquid templating and modular Web Components to lay out the structure of your page based on incoming task data, or
2. Minimize the role of templating, using Liquid to pass in more complex data objects and handling layout mainly in a more monolithic, Vue-based context.

But Vue.js, like other frameworks, brings specific and at times opinionated patterns for structuring the flow of data within and between components in an app - which may not align exactly with native DOM and Web Component patterns.

Because the Vue component is composed at a higher level of abstraction than the DOM CustomElement it's bound to with `vue.defineCustomElement` and `window.customElements.define`, workarounds can be required for tasks like:

- Emitting events from a Vue Web Component that bubble up through the DOM tree (see [MultiFieldValue.ce.vue](src/components/MultiFieldValue.ce.vue))
- Exposing functional interfaces on the Element itself that parents or other components can call on-demand (see `validate()` in [MultiFieldValue.ce.vue](src/components/MultiFieldValue.ce.vue))
- Correctly handling insertion, update and deletion of child components which might normally be handled in the framework by simply [binding](https://vuejs.org/guide/essentials/list.html#v-for) to a reactive list variable (see list/add/remove/watch logic in [FieldMultiValue.ce.vue](src/components/FieldMultiValue.ce.vue))

In other words, the more you modularize to separate native Custom Elements and complex Liquid templating (option 1), the further your code will likely drift from normal Vue app patterns. Eventually, the value of using Vue would likely diminish to not being worth the cost of managing these alignment gaps (maybe you could even start writing native Custom Elements instead?).

By pushing a bit more of the functional complexity into consolidated Vue components (or even going as far as a monolithic `<div id="app">`), you'll likely get more benefit from the framework at the expense of less templating flexibility.


### Building in SageMaker (Studio)

As shown in the [accompanying notebook](../3.%20Human%20Review.ipynb), you can certainly build and preview the template from inside SageMaker Studio/notebooks. But what about using these JupyterLab-based environments for significant template development/extension work?

There are some factors you'll want to consider (correct at time of writing):

1. The code in this sample/folder uses 2-space indentation (common in JS and web projects), while the default JupyterLab indentation is 4 spaces (common in Python projects).
  - You can change indentation in the bottom right of the file editor (button showing `Spaces: 4` by default) or through `Settings > Advanced Settings Editor > Text Editor`. However, this setting is global - so you'll want to switch it back to 4 spaces when returning to editing Python or notebook files instead.
2. Remember if using a system terminal for your `npm` commands in SageMaker Studio, that *system terminals* run on the environment that hosts your JupyterLab instance, which is **separate from** the environments for your *notebook kernels* or "image terminals".
  - A version of NodeJS will already be installed in the system environment and you may face unexpected errors if you try to overwrite/upgrade the system NodeJS version. At the time of writing, SM Studio used NodeJS v12 which was also compatible with this Vue project.
3. If running the Vite dev server `npm run dev`, you'll need to connect through SageMaker port proxy path.
  - To access a Vite dev server run from a system terminal with `npm run dev`, instead of `http://localhost:3000/index-noliquid.html` you would connect to your own domain's address with a port-forwarding proxy path - something like: `https://{DOMAIN_ID}.studio.{AWS_REGION}.sagemaker.aws/jupyter/default/proxy/3000/index-noliquid.html`
  - This setup will likely need some additional [configuration](https://vitejs.dev/config/) of Vite in [vite.config.js](vite.config.js), to ensure the client fetches assets under the `/jupyter/default/proxy/3000/` path while the server receives the requests as if they're from the root path `/`.

Alternatively to avoid tackling these issues, you could use a local IDE for interactive template development with the `npm run dev` live-updating dev server - and then upload the edited source to SageMaker to run final tests in the A2I context.

> **Remember:** You can use the dev server and the alternative `index-noliquid.html` page to iteratively build and test UI functionality with placeholder fulfilled template values, without having to go through the full `smclient.render_ui_template(...)` flow to see every change.
