# berryboxai

## Installation

1. Install Conda (if not already installed):

    a. If possible, install conda from [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.anaconda.com/miniconda/)
    
    b. If using a USDA-ARS machine, use the following instructions:
    1. Open the "Software Center" app (Windows) or "Self Service" app (MacOS)
    2. Search for "Anaconda" and click "Install." Note that each user must install Anaconda separately.
    3. Wait for installation to complete (it may take a while).

2. Clone the repository:
    + Option 1: Use Git
        + If using a Windows machine, open up Powershell and run the following command:
            ```powershell
            cmd
            ```
        + If using a Mac, simply open up Terminal.
        + Navigate to the Documents folder (or your preferred destination):  
            + On a PC, use:
                ```
                cd C:/Users/[USERNAME]/Documents
                ```
            + On a Mac, use:
                ```
                cd /Users/[USERNAME]/Documents
                ```
                > Remember to replace `[USERNAME]` with your actual username.

        + Run the following command:
            ```
            git clone https://github.com/NeyhartLab/berryboxai.git
            ```

    + Option 2: Download the repository as a .zip file
        + [Download the .zip file](https://github.com/NeyhartLab/berryboxai/archive/refs/heads/main.zip)
        + Move the .zip file to a suitable location, like the Documents folder.
        + Unzip the .zip file
        + Open up Powershell or Terminal and navigate to the Documents folder:
            + On a PC, use:
                ```
                cmd
                cd C:/Users/[USERNAME]/Documents
                ```
            + On a Mac, use:
                ```
                cd /Users/[USERNAME]/Documents
                ```
                > Remember to replace `[USERNAME]` with your actual username.

3. While Powershell or Terminal is still open, nagivate to the `berryboxai` folder:
    ```
    cd berryboxai
    ```

4. Create a Conda environment by typing the following:
    ```
    conda env create -f environment.yml
    ```

5. Activate the environment:
    ```
    conda activate berryboxai_env
    ```

6. Install the package
    ```
    python -m pip install .
    ```

7. If the installation was successful, you should be able to call the `berryboxai` script directly from the command line within Powershell or Terminal:
    ```
    berryboxai --help
    ```


## Usage

### Follow the below instructions for routine use.

1. Open Powershell or Terminal

2. Activate the conda environment:
    ```
    conda activate berryboxai_env
    ```

3. Run `berryboxai` for the desired use or module

### Modules

Currently there are two modules: `berry-seg` or `rot-det`:
+ `berry-seg` is used when measuring shape, size, color attributes on sound fruit
+ `rot-det` is used when determining the level of rot in a sample directly from the field.

### Interactive mode

Interactive mode uses a microcontroller (raspberry pi) to control the camera and a barcode scanner to record samples

```
berryboxai -m [module] --output /path/to/output --save --preview
```
> Remember to replace `[module]` with the module name (`berry-seg` or `rot-det`) and replace `/path/to/output` with the path to the desired output folder

The `--save` option saves images plus the predictions within the output folder.  
The `--preview` option displays a preview of the image plus the predictions.

### Batch mode

Batch mode is used when you already have a folder of images and you want to run the berrybox model on those images to generate predictions

```
berryboxai -m [module] --input /path/to/image/folder --output /path/to/output --save
```
> Remember to replace `[module]` with the module name (`berry-seg` or `rot-det`), replace `/path/to/output` with the path to the desired output folder, and `/path/to/image/folder` with the path to the folder containing the images you want to run through the prediction model.

The `--input` option is used to control whether you want interactive mode or batch mode. If you provide a path in `--input`, the script will assume that you want to run batch mode.  
In batch mode, the `--preview` option is disabled.



### Shiny App

The command line tools have been wrapped into a Shiny App. To launch it, after loading the conda environment, run

`berryboxaiApp`

from the command line.