# MBRL-Lib Diagnostics and Visualization tools
This package contains a set of components that can be useful for debugging and
visualizing the operation of your models and controllers. Currently, 
the following tools are provided:

* ``Visualizer``: Creates a video to qualitatively
assess model predictions over a rolling horizon. Specifically, it runs a 
  user specified policy in a given environment, and at each time step, computes
  the model's predicted observation/rewards over a lookahead horizon for the 
  same policy. The predictions are plotted as line plots, one for each 
  observation dimension (blue lines) and reward (red line), along with the 
  result of applying the same policy to the real environment (black lines). 
  The model's uncertainty is visualized by plotting lines the maximum and 
  minimum predictions at each time step. The model and policy are specified 
  by passing directories containing configuration files for each; they can 
  be trained independently. The following gif shows an example of 200 steps 
  of pre-trained MBPO policy on Inverted Pendulum environment.
  \
  \
  ![Example of Visualizer](http://raw.githubusercontent.com/facebookresearch/mbrl-lib/main/docs/resources/inv_pendulum_mbpo_vis.gif)
  <br>
  <br>
* ``DatasetEvaluator``: Loads a pre-trained model and a dataset (can be loaded from separate directories), 
  and computes predictions of the model for each output dimension. The evaluator then
  creates a scatter plot for each dimension comparing the ground truth output 
  vs. the model's prediction. If the model is an ensemble, the plot shows the
  mean prediction as well as the individual predictions of each ensemble member.
  \
  \
  ![Example of DatasetEvaluator](http://raw.githubusercontent.com/facebookresearch/mbrl-lib/main/docs/resources/dataset_evaluator.png)
  <br>
  <br>
* ``FineTuner``: Can be used to train a model on a dataset produced by a given agent/controller. 
  The model and agent can be loaded from separate directories, and the fine tuner will roll the 
  environment for some number of steps using actions obtained from the 
  controller. The final model and dataset will then be saved under directory
  "model_dir/diagnostics/subdir", where `subdir` is provided by the user.\
  <br>
* ``True Dynamics Multi-CPU Controller``: This script can run
a trajectory optimizer agent on the true environment using Python's 
  multiprocessing. Each environment runs in its own CPU, which can significantly
  speed up costly sampling algorithm such as CEM. The controller will also save
  a video if the ``render`` argument is passed. Below is an example on 
  HalfCheetah-v2 using CEM for trajectory optimization. To specify the environment,
  follow the single string syntax described 
  [here](https://github.com/facebookresearch/mbrl-lib/blob/main/README.md#supported-environments).
  \
  \
  ![Control Half-Cheetah True Dynamics](http://raw.githubusercontent.com/facebookresearch/mbrl-lib/main/docs/resources/halfcheetah-break.gif)
  <br>
  <br>
* [``TrainingBrowser``](training_browser.py): This script launches a lightweight
training browser for plotting rewards obtained after training runs 
  (as long as the runs use our logger). 
  The browser allows aggregating multiple runs and displaying mean/std, 
  and also lets the user save the image to hard drive. The legend and axes labels
  can be edited in the pane at the bottom left. Requires installing `PyQt5`. 
  Thanks to [a3ahmad](https://github.com/a3ahmad) for the contribution.
  \
  \
  ![Training Browser Example](http://raw.githubusercontent.com/facebookresearch/mbrl-lib/main/docs/resources/training-browser-example.png)

Note that, except for the training browser and the CPU-controller, all the tools above require Mujoco 
installation and are specific to models of type 
[``OneDimTransitionRewardModel``](../models/one_dim_tr_model.py).
We are planning to extend this in the future; if you have useful suggestions
don't hesitate to raise an issue or submit a pull request!