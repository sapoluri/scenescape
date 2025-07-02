# Getting Started with Intel® SceneScape

-   **Time to Complete:** 30-45 minutes

## Get Started

### Prerequisites

* To run and interact with the demo scenes that the Intel® SceneScape release package provides, one computer is needed. The ./deploy.sh script has been tested to work with Ubuntu* 22.04 Desktop.
* The computer must be at least a 10th Generation Intel® Core™ i5 Processor or Intel® Xeon® Scalable processor, with at least 8+GB of RAM and 64+GB of storage. This configuration is the minimum to run the out of the box demos and more compute resources maybe be required for additional models, cameras, and/or sensors.
* For the initial build of the ./deploy.sh process the computer must have a correctly configured connection to the Internet for acquiring the needed build tools. When using a proxy, the proxy will need to be correctly configured for the console environment, OS package installer, and Docker*. Deployed containers can run without an internet connection.
* When deploying a live scene, a scale floor plan of the scene is needed either in a 3D scene scan in .glb format or in a 2D web image format (JPG, PNG, or GIF) that is about 600 to 1000 pixels wide. Walls and fiducial markers on the floor plan must be at least twice as accurate as the desired tracking accuracy (e.g. accuracy < 1 meter requires a floor plan accurate to < 0.5 meters).
* It is not recommended to initially use a virtual machine. Once Intel® SceneScape is configured for a specific use case and the system resource requirements are understood, then a multicore VM could be configured for deployment and execution. Windows Subsystem for Linux* (WSL) is not supported.

### Step 1: Install OS

Follow this tutorial to download Ubuntu 22.04 and install it on the target computer selecting the minimal installation option and to erase the disk and install Ubuntu: https://ubuntu.com/tutorials/install-ubuntu-desktop
After the install be sure to update the system software before proceeding.
```console
sudo apt update
```

### Step 2: Get SceneScape sources

**Note:** These operations must be executed when logged in as a standard (non-root) user. **Do NOT use root or sudo.**
#### Option 1: Download and extract the sources

1.  Download the Intel® SceneScape software archive from https://github.com/open-edge-platform/scenescape/releases.

2.  Extract the Intel® SceneScape archive on the target Ubuntu 22.04 system. Change directories to the extracted Intel® SceneScape folder.
    ```bash
    cd scenescape-<version>/
    ```

#### Option 2: Clone the repository from GitHub

If you prefer to work directly with the source repository instead of downloading a release archive, follow these steps:

1. Install Git if it's not already installed:
  ```bash
  sudo apt install git -y
  ```

2. Clone the SceneScape repository:
  ```bash
  git clone https://github.com/open-edge-platform/scenescape.git
  ```

3. Change directories to the cloned repository:
  ```bash
  cd scenescape/
  ```

4. Check out the desired version tag (e.g., v1.3.0):
  ```bash
  git checkout v1.3.0
  ```

### Step 3: Build and deploy Intel® SceneScape to the target system

The deployment tool may take around 30 minutes to download and build the Intel® SceneScape container images. Do not disconnect the system from the network or close the terminal window during this process.

* **Note:** The deployment script will ask for a SUPASS for logging into Intel® SceneScape, Important: This should be different
than the password for your system user. Be sure to enter the same password twice to verify:
    ```console
    Enter SUPASS:
    Verify:
    ```
* **Note:** If this is the first time deploying Intel® SceneScape on this system, the system will prompt to enter the user’s sudo password to enable installation of host software packages like Docker. This password is not stored by Intel® SceneScape or used outside of installing the required software.
    ```console
    [sudo] password for <username>:
    ```
*  Begin the install using
    ```console
    ./deploy.sh
    ```
* **Note:** Depending on what is installed on the system, various prompts may ask to enter the user’s sudo password again or press Enter to continue.

The deployment process will take some time as dependent components are downloaded, integrated, and tested. When `deploy.sh` completes successfully, the system will be running.

**Note:** The deploy command is intended to be run only once. In order to stop/re-start the system, follow the instructions in 'Stopping the system' and 'Starting the system'.

### Step 4: Verify a successful deployment

If you are running remotely, connect using ```"https://<ip_address>"``` or ```"https://<hostname>"```, using the correct IP address or hostname of the remote Intel® SceneScape system. If accessing on a local system use ```"https://localhost"```. If you see a certificate warning, click the prompts to continue to the site. For example, in Chrome click "Advanced" and then "Proceed to &lt;ip_address> (unsafe)".

> **Note:** These certificate warnings are expected due to the use of a self-signed certificate for initial deployment purposes. This certificate is generated at deploy time and is unique to the instance.

### Logging In
Enter "admin" for the user name and the value you typed earlier for SUPASS.

### Stopping the System

To stop the containers, use the following command in the project directory:

```console
$ docker compose down --remove-orphans
```
### Starting the System

To start after the first time, use the following command in the project directory:

```console
$ docker compose up -d
```

## Summary

Intel® SceneScape was downloaded, built and deployed onto a fresh Ubuntu 22.04 system. Using the web user interface, Intel® SceneScape provides two scenes by default that can be explored running from stored video data.
![SceneScape WebUI Homepage](images/homepage.png)
* **Note** the “Documentation” menu option, click to view the Intel® SceneScape HTML version of the documentation in the browser.

## Next Steps
- **How to enable reidentification**
  - [How to enable reidentification](How-to-enable-reidentification.md): Step-by-step guide to enable reidentification.

- **How to use sensor types**
  - [How to use Sensor types](How-to-use-sensor-types.md): Step-by-step guide to getting started with sensor types.

- **How to use 3D UI**
  - [How to use 3D UI](How-to-use-3D-UI.md): A guide on how use 3D UI

- **How to create a Geti trained AI models and integrate it with Intel® SceneScape.**
  - [Geti AI model integration](How-to-integrate-geti-trained-model.md): Step-by-step guide for integrating a Geti trained AI model with Intel® SceneScape.

- **How to visualize regions**
  - [How to visualize regions](How-to-visualize-regions.md): Step-by-step guide to getting started with visualizing regions.

- **How to configure a hierarchy of scenes**
  - [How to configure a hierarchy of scenes](How-to-configure-a-hierarchy-of-scenes.md): Step-by-step guide to configuring a hierarchy of scenes.

- **How to manually calibrate cameras**
  - [How to manually calibrate cameras](How-to-manually-calibrate-cameras.md): Step-by-step guide to performing Manual Camera Calibration.

- **How to autocalibrate cameras using visual features**
  - [How to autocalibrate cameras using visual features](How-to-autocalibrate-cameras-using-visual-features.md): Step-by-step guide to performing Auto Camera Calibration using Visual Features.

- **How to autocalibrate cameras using Apriltags**
  - [How to autocalibrate cameras using Apriltags](How-to-autocalibrate-cameras-using-apriltags.md): Step-by-step guide to performing Auto Camera Calibration using Apriltags.

- **How to upgrade Intel® SceneScape**
  - [How to upgrade Intel Scenescape](How-to-upgrade.md): Step-by-step guide for upgrading from an older version of Intel® SceneScape.

- **How to inference using Nvidia GPU with OVMS**
  - [How to inference using Nvidia GPU with OVMS](How-to-inference-using-Nvidia-gpu-with-OVMS.md): Step-by-step guide for enabling inference on Nvidia GPU using OVMS.

## Learn More

-   Understand the components, services, architecture, and data flow, in
    the [Overview](Overview.md).
-   Follow examples to become familiar with the core functionality of Intel® SceneScape, in
    [Tutorial](Tutorial.md).
-   Optimizing security posture for a Intel® SceneScape installation [Hardening Guide for Custom TLS](hardening-guide.md)
