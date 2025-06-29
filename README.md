# ProfilePlotter


## Description
This was developed from Paul McLachlan's Profiler code when he was at the HydroGeophysics Group, Aarhus University. The name has been change to ProfilePlotter for not creating confusion with TEMCompany's Profiler instrument


## Installation

### 📦 Clone the repository
#### Option 1: GitHub (HTTPS)

git clone https://github.com/yourusername/profileplotter.git
cd profileplotter
#### Option 2: GitLab (SSH, for internal collaborators)
git clone git@gitlab.au.dk:hgg/profileplotter.git
cd profileplotter


### ⚙️ Set up the environment (recommended:Conda)

#use the yaml
conda env create -f profplotter.yml

#### Activate it
conda activate profplotter

#### (Optional but recommended)
pip install -e .


### 🌍 Try and read one of the examples
Go to the Examples folder. It's easier to follow with a Jupyter notebook

#### Run the Kakuma.ipynb

### 🧱👷‍♂️ Create your own!

### Profiles without tTEM data

`createProfiles` can now be called with `ttem_model_idx=None` to generate the
profile geometry using only sTEM or Profiler soundings. This allows projecting
these soundings along a profile even when no tTEM models are available within
the search radius.
