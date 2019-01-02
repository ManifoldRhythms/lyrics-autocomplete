# Generate lyrics based on a Tensorflow model.

<walkthrough-tutorial-duration duration="10"></walkthrough-tutorial-duration>

## Let's get started!

This tutorial demonstrates using Cloud TPUs to build a _language model_: a model that predicts the next character of 
text given the text so far.

Once our model has been trained we can sample from it to generate new text that looks like 
the text it was trained on. In this case we're going to train our network using the combined works of Shakespeare, 
creating a play generating robot.

<walkthrough-footnote>
Note: You will need a GCP account and a GCS bucket for this notebook to run!

Click the **Next** button to move to the next step.
</walkthrough-footnote>

## Activate a GCP Project

<walkthrough-project-setup></walkthrough-project-setup>

## Create a Compute Engine VM and a Cloud TPU

### Compute Instance Configuration:

* 2 vCPUs
* 7.5 GB memory
* 250 GB standard persistent disk
* 1 Cloud TPU Preemptible v2

The above configuration will cost about $1.445/hour. ([TPU pricing][tpu_pricing])

### Create the instance

<walkthrough-watcher-constant key="instance-name" value="lyrics-tpu">
</walkthrough-watcher-constant>

<walkthrough-watcher-constant key="instance-zone" value="us-central1-b">
</walkthrough-watcher-constant>

<open-cloud-shell-button open-cloud-shell devshell-precreate></open-cloud-shell-button>

```bash
ctpu up -preemptible -preemptible-vm \
-name {{instance-name}} \
-project {{project-id}} \
-zone {{instance-zone}}
```

You should see a message like this:

```
jeremy@cloudshell:~ (mr-lyrics-autocomplete)$ ctpu up -preemptible -preemptible-vm -name {{instance-name}}
ctpu will use the following configuration:

  Name:                 {{instance-name}}
  Zone:                 {{instance-zone}}
  GCP Project:          {{project-id}}
  TensorFlow Version:   1.12
  VM:
      Machine Type:     n1-standard-2
      Disk Size:        250 GB
      Preemptible:      true
  Cloud TPU:
      Size:             v2-8
      Preemptible:      true

OK to create your Cloud TPU resources with the above configuration? [Yn]:
```

**Tip**: Click the copy button on the side of the code box to paste the command in the Cloud Shell terminal to run it.


## Clone model

### Verify your Compute Engine VM
When the `ctpu up` command has finished executing, verify that your shell prompt has changed from `username@cloudshell` 
to `username@{{instance-name}}`. This change shows that you are now logged into your Compute Engine VM.

### Clone git repo

```bash
git clone https://github.com/ManifoldRhythms/lyrics-autocomplete
&& cd lyrics-autocomplete
```

### Setup Environment

```bash
npm install && ./cli.js storage setup
```

### Verify the data file

```bash
gsutil ls gs://mr-lyrics-autocomplete-data/lyrics_data.txt
```

## Training our model

Since we're using TPUEstimator, we need to provide what's called a _model function_ to train our model. 
This specifies how to train, evaluate and run inference (predictions) on our model.

Let's cover each part in turn. We'll first look at the training step.

* We feed our source tensor to our LSTM model
* Compute the cross entropy loss to train it better predict the target tensor.
* Use the `RMSPropOptimizer` to optimize our network
* Wrap it with the `CrossShardOptimizer` which lets us use multiple TPU cores to train.

Finally we return a `TPUEstimatorSpec` indicating how TPUEstimator should train our model.

<walkthrough-editor-select-line filePath="lyrics-autocomplete/tf/lstm_model.py" startLine="125" startCharacterOffset="0" endLine="125" endCharacterOffset="30">
Open `train_fn`
</walkthrough-editor-select-line>


## Evaluating our model

Next, evaluation. This is simpler: we run our model forward and check how well it predicts the next character. 
Again, we return a `TPUEstimatorSpec` to tell TPUEstimator how to evaluate the model.

<walkthrough-editor-select-line filePath="lyrics-autocomplete/tf/lstm_model.py" startLine="143" startCharacterOffset="0" endLine="143" endCharacterOffset="29">
Open `eval_fn`
</walkthrough-editor-select-line>


## Computing Predictions

We leave the most complicated part for last. There's nothing TPU specific here! For predictions we use the input tensor 
as a seed for our model. We then use a TensorFlow loop to sample characters from our model and return the result.

<walkthrough-editor-select-line filePath="lyrics-autocomplete/tf/lstm_model.py" startLine="164" startCharacterOffset="0" endLine="164" endCharacterOffset="24">
Open `predict_fn`
</walkthrough-editor-select-line>


## Building our model function

We can now use our helper functions to build our combined model function and train our model!

<walkthrough-editor-select-line filePath="lyrics-autocomplete/tf/lstm_model.py" startLine="220" startCharacterOffset="0" endLine="220" endCharacterOffset="46">
Open `model_fn`
</walkthrough-editor-select-line>


## Tensorboard

<walkthrough-spotlight-pointer cssSelector=".p6n-devshell-add-tab-button">Open</walkthrough-spotlight-pointer>
a second Cloud Shell session to run TensorBoard.

Once the session is open, ssh into your training vm with port forwarding. This will allow you to view 
Tensorboard using the Cloud Shell web preview.

```bash
gcloud compute ssh {{instance-name}} \
--project {{project-id}} \
--zone {{instance-zone}}
--ssh-flag=-L8080:localhost:8080
```

Then start Tensorboard

```bash
tensorboard --port 8080 --logdir=gs://mr-lyrics-autocomplete-data/model/log
```

## Running our model

Now we can train our model!

### Train Length

We will train for
<walkthrough-editor-select-line filePath="lyrics-autocomplete/tf/lstm_model.py" startLine="10" startCharacterOffset="10" endLine="10" endCharacterOffset="15">
2000
</walkthrough-editor-select-line>
steps.

**Tip**: You may change this number to increase or decrease the amount of steps the model will train.

### Start Training

Using the 
<walkthrough-spotlight-pointer cssSelector=".p6n-devshell-tab-bar .goog-tab:first-child">first</walkthrough-spotlight-pointer>
Cloud Shell session start the training.

```bash
python3 tf/lstm_model.py
```

### Monitor Training

Now you can open the
<walkthrough-spotlight-pointer spotlightId="devshell-web-preview-button">web preview</walkthrough-spotlight-pointer>
to view the progress in Tensorboard.


## Make Predictions




## Preemptible TPUs and VM Instances

A preemptible TPU is a Cloud TPU node that you can create and run at a much lower price than normal nodes. 
However, Cloud TPU may terminate (preempt) these nodes if it requires access to the resources for another purpose.

Restarting a pre-emptible instance after a shutdown is a 1-click operation from the Google management
page (or one command using Google Cloud Shell).

<walkthrough-footnote>

You can also restart your VM instance with one click from your smartphone using the [Google Console app](https://cloud.google.com/console-app/).

**Reference Links**

* [Preemptible VM Instances][gce_preemptible_docs]
* [Using Preemptible TPUs][tpu_preemptible_docs]

</walkthrough-footnote>

## Congratulations

<walkthrough-conclusion-trophy></walkthrough-conclusion-trophy>

You're all set!


[gcp_tpu_example]: https://github.com/tensorflow/tpu/blob/master/tools/colab/shakespeare_with_tpuestimator.ipynb
[tpu_pricing]: https://cloud.google.com/tpu/docs/pricing#pricing_example_using_a_preemptible_tpu
[tpu_preemptible_docs]: https://cloud.google.com/tpu/docs/preemptible
[gce_preemptible_docs]: https://cloud.google.com/compute/docs/instances/preemptible
