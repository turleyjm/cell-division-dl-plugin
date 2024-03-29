# cell-division-dl-plugin

Deep learning plugin to detect cell divisions and find there orientation. 

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

Install by cloning this repository, and once inside this repository make a new virtual environment and activate it by running:

For mac
```sh
python3 -m venv venv
source venv/bin/activate
```
For windows
```sh
python3 -m venv venv
.\venv\Scripts\activate
```

Next install the napari and plugin
```sh
pip install git+https://github.com/turleyjm/cell-division-dl-plugin.git
```

Models (UNetCellDivision10.pth.tar and UNetOrientation.pth.tar) needed to run this plugin can be downloaded from [Zenodo]. Add them to models folder.

(Note requires version >=python3.8)

## Usage

Loading napari can be done by typing `napari` in your terminal with in the virtual environment this will load the napari GUI. Then the video cell divisions are to be detected in can be dragged and dropped in to the GUI. We have added a demo video (focusWoundL18h10-short.tif) to the in the dat folder which can used to demonstrate the method. The plugin can be then be started.

After loading the plugin clicking the button "" there are 3 options for "Division heatmap", "Division database" and "Division & orientaton database" these will each produce a different a

### Division heatmap
"Division heatmap" loads and runs the UNetCellDivision10.pth.tar model on the video. It displays the division prediction heatmap showing areas of the video it has detected cell divisions. (Note some of the output signal can be weak in some division and is still detected in the next steps but is by eye more difficult to see. Increasing the bightness/contrast can clearly show these detected divisons)

![displayDivisions](https://github.com/turleyjm/cell-division-dl-plugin/assets/68009166/56dba27c-7c78-4231-b574-61294cec449a)

### Division database
"Division database" will do the first step as above and also locate these divisions from the prediction heatmap and genate a database of cell divisions finding their location in space and time. Also saves a display video in the folder "output" showing detected cell divisions via a blue overlay.

### Division & orientaton database
"Division & orientaton database" follows the same steps as above then loads and runs the UNetOrientation.pth.tar model on each of the detected cell divisions to determine the orientation of cell divisions and saves this in a updated database. Also saves a display video in the folder "output" showing detected cell divisions and their orientation via a blue overlay.

Extracting divisions from the video can take some time to process. As 5 frame videos are used to detect cell divisions, cells dividing in the first or last 3 frames will not have their cell divisions detected.

## Retrain 

This model could be used for other 2D cell division data but retraining is likely to be needed for highly accurate models. Here we have used 2 flourencent channels showing cell nuclei and boundaries to supply dynamic information to the model. Other systems may use different markers to label these parts of cell or used different componds altogether (e.g microtubules).

In the folder trainModels we have included the code to train, test and evaluate each of the deep learning models we have used in the paper. The datasets used to train the models can be found in [Zenodo]. Also included is the saved parameters of the trained models.

To retrain a model on a new set of exprimental data can be done by genarating a set of training data following the examples provide in the training data. Then using the proved Jupyter notebooks to train the indivual models. For those less familiar with training deep learning model we recommend the freely avalable [fast.ai] training course.

## Expirmental data and analysis

The data used in the analysis shown in are [paper] can be downloaded from [Zenodo]. The scripts in the folder analysis process the databases and make the figures of the paper. Also the files called "focus{sampleName}.tif" and similar videos can be used as inputs for the plugin.

## License

Distributed under the terms of the [BSD-3] license,
"cell-division-dl-plugin" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin
[paper]: https://www.biorxiv.org/content/10.1101/2023.03.20.533343v3.abstract
[Zenodo]: https://zenodo.org/records/10846684
[fast.ai]: https://course.fast.ai/Lessons/lesson1.html


[file an issue]: https://github.com/turleyjm/cell-division-dl-plugin/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
