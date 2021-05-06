# OpenAI Gym wrapper for the DeepMind Control Suite.
A lightweight wrapper around the DeepMind Control Suite that provides the standard OpenAI Gym interface. The wrapper allows to specify the following:
* Reliable random seed initialization that will ensure deterministic behaviour.
* Setting ```from_pixels=True``` converts proprioceptive observations into image-based. In additional, you can choose the image dimensions, by setting ```height``` and ```width```.
* Action space normalization bound each action's coordinate into the ```[-1, 1]``` range.
* Setting ```frame_skip``` argument lets to perform action repeat.


### Instalation
```
pip install git+git://github.com/denisyarats/dmc2gym.git
```

### Usage
```python
import dmc2gym

env = dmc2gym.make(domain_name='point_mass', task_name='easy', seed=1)

done = False
obs = env.reset()
while not done:
  action = env.action_space.sample()
  obs, reward, done, info = env.step(action)
```
