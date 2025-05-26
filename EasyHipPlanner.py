# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 08:42:14 2024

@author: przem
"""

import slicer
import vtk
import qt
import os
import math
from DICOMLib import DICOMUtils
import numpy as np

# Global variables for segmentation and transform nodes
L_Cup_Node = None
R_Cup_Node = None
L_Tr_Node = None
R_Tr_Node = None
Head_Node = None

L_Cup_Transform = None
R_Cup_Transform = None
L_Tr_Transform = None
R_Tr_Transform = None

Head_Transform_L = None  # Transform for left cup
Head_Transform_R = None  # Transform for right cup
Head_Visible = False
distanceLabel = None

initialSceneState = None
xrayTransformMatrix = None

currentSegmentationNode = None
currentSegmentID = None
originalPinSize = None  # Wyjściowy rozmiar
currentPinSize = None   # Aktualny rozmiar
maxPinSize = None       # Maksymalny rozmiar

currentXRayScale = 115  # Domyślna wartość po imporcie

# Globalne zmienne dla przycisków referencyjnych
leftReferenceButton = None
rightReferenceButton = None

# Globalne zmienne dla przesunięcia pionowego i długości poziomej
referenceLineOffset = 35  # Domyślne przesunięcie w mm (pionowe)
horizontalLineLength = 35  # Domyślna długość linii poziomej w mm

rotationalCorrectionEnabled = False 

left_vertical_shift = None
right_vertical_shift = None

# Przyciski i checkboxy
L_Tr_Checkbox = None
R_Tr_Checkbox = None
L_Cup_Checkbox = None
R_Cup_Checkbox = None
Ischial_Checkbox = None
referenceLineButton = None

# Utility functions
def isVolumeLoaded():
    """
    Check if a volume is loaded in the scene.
    """
    try:
        slicer.util.getNode("vtkMRMLScalarVolumeNode*")
        return True
    except slicer.util.MRMLNodeNotFoundException:
        return False

def reset2DAnd3DViews():
    """
    Resetuje widoki 2D i 3D, aby dopasować je do zawartości.
    """
    layoutManager = slicer.app.layoutManager()
    if layoutManager:
        # Reset widoków 3D
        for viewIndex in range(layoutManager.threeDViewCount):
            threeDWidget = layoutManager.threeDWidget(viewIndex)
            threeDView = threeDWidget.threeDView()
            threeDView.resetFocalPoint()

        # Reset pola widzenia we wszystkich widokach 2D (przekroje)
        for sliceViewName in layoutManager.sliceViewNames():
            sliceWidget = layoutManager.sliceWidget(sliceViewName)
            sliceWidget.sliceLogic().FitSliceToAll()
            
    
    
def validateTransformAssignments():
    """
    Validate that all relevant segmentation nodes are assigned to the correct transform.
    """
    transformNodes = {
        "Right Tr Segmentation": R_Tr_Transform,
        "Right Cup Segmentation": R_Cup_Transform,
    }
    for segmentationName, transformNode in transformNodes.items():
        try:
            segmentationNode = slicer.util.getNode(segmentationName)
            if not segmentationNode:
                print(f"[Debug] Segmentation node '{segmentationName}' not found.")
                continue
            if transformNode and segmentationNode.GetParentTransformNode() != transformNode:
                segmentationNode.SetAndObserveTransformNodeID(transformNode.GetID())
                print(f"[Debug] Assigned transform to {segmentationName}: {transformNode.GetName()}")
        except Exception as e:
            print(f"[Error] Failed to assign transform for {segmentationName}: {e}")

def getTransformPosition(segmentationNode):
    """
    Get the global position of a segment's parent transform.

    :param segmentationNode: The segmentation node containing the segment.
    :return: A list [x, y, z] representing the global position of the transform.
    """
    parentTransformNode = segmentationNode.GetParentTransformNode()
    if not parentTransformNode:
        print(f"[Transform Debug] No transform node found for segmentation '{segmentationNode.GetName()}'.")
        return [0.0, 0.0, 0.0]

    transformMatrix = vtk.vtkMatrix4x4()
    parentTransformNode.GetMatrixTransformToWorld(transformMatrix)

    # Extract translation from the transform matrix
    position = [
        transformMatrix.GetElement(0, 3),  # X translation
        transformMatrix.GetElement(1, 3),  # Y translation
        transformMatrix.GetElement(2, 3),  # Z translation
    ]
    print(f"[Debug] Transform position for '{segmentationNode.GetName()}': {position}")
    return position

def getSegmentCenter(segmentationNode, segmentID):
    """
    Get the center of a segment based on its initial transform or position.

    :param segmentationNode: The segmentation node containing the segment.
    :param segmentID: The ID of the segment.
    :return: A list [x, y, z] representing the center of the segment in global RAS coordinates.
    """
    if not segmentationNode or not segmentID:
        print("[Segment Error] Invalid segmentation node or segment ID.")
        return [0.0, 0.0, 0.0]

    # Retrieve the position of the segment from the transform or as a fallback use a default
    transformNode = segmentationNode.GetParentTransformNode()
    if transformNode:
        # Use the transform's origin or pivot point as the center
        transformMatrix = vtk.vtkMatrix4x4()
        transformNode.GetMatrixTransformToWorld(transformMatrix)

        # Extract translation (last column of the matrix)
        globalCenter = [
            transformMatrix.GetElement(0, 3),
            transformMatrix.GetElement(1, 3),
            transformMatrix.GetElement(2, 3)
        ]
        print(f"[Debug] Segment '{segmentID}' center based on transform: {globalCenter}")
        return globalCenter

    # If no transform is applied, fall back to default or predefined position
    print(f"[Debug] No transform node found. Using default position for segment '{segmentID}'.")
    return [0.0, 0.0, 0.0]  # Możesz tu wstawić bardziej sensowną domyślną wartość, jeśli jest znana.



# Global variables for X-ray mode state
XRay_Mode_Enabled = False
XRay_Transform_Node = None

# Global variable to store the original layout
originalLayout = None

initialSceneState = None  # Global variable to store the initial scene state

sceneSnapshot = None  # Global variable to store the scene snapshot

sceneSnapshotPath = None  # Global variable to store the path to the scene snapshot file

def importDICOM():
    """
    Import DICOM data (single file or series) into 3D Slicer, resetting global variables and scene state.
    """
    global dicomFilePaths, loadedNode, isSingleDICOM
    global L_Cup_Node, R_Cup_Node, L_Tr_Node, R_Tr_Node, Head_Node
    global L_Cup_Transform, R_Cup_Transform, L_Tr_Transform, R_Tr_Transform
    global Head_Transform_L, Head_Transform_R, Head_Visible, distanceLabel
    global currentSegmentationNode, currentSegmentID, originalPinSize, currentPinSize, maxPinSize
    global Ischial_Line_Node, XRay_Transform_Node, XRay_Mode_Enabled
    global cupSizeSpinBoxL, cupSizeSpinBoxR

    print("[Debug] Starting DICOM import.")

    # Reset global variables
    L_Cup_Node = R_Cup_Node = L_Tr_Node = R_Tr_Node = Head_Node = None
    L_Cup_Transform = R_Cup_Transform = L_Tr_Transform = R_Tr_Transform = None
    Head_Transform_L = Head_Transform_R = None
    Head_Visible = False
    distanceLabel = None
    currentSegmentationNode = currentSegmentID = None
    originalPinSize = currentPinSize = maxPinSize = None
    Ischial_Line_Node = XRay_Transform_Node = None
    XRay_Mode_Enabled = False
    dicomFilePaths = []
    loadedNode = None
    isSingleDICOM = False

    # Clear the scene
    slicer.mrmlScene.Clear(0)
    print("[Debug] Cleared the scene for new DICOM import.")

    # Reset spinbox values for cups
    for spinBox, label in [(cupSizeSpinBoxL, "left"), (cupSizeSpinBoxR, "right")]:
        if spinBox is not None:
            spinBox.blockSignals(True)
            spinBox.setValue(50)
            spinBox.blockSignals(False)
            print(f"[Debug] Reset {label} cup size spinbox to 50.")

    # Open file dialog to select DICOM files
    fileDialog = qt.QFileDialog()
    fileDialog.setFileMode(qt.QFileDialog.ExistingFiles)
    fileDialog.setNameFilter("DICOM Files (*.dcm);;All Files (*)")
    fileDialog.setWindowTitle("Import DICOM Files")

    if fileDialog.exec_() == qt.QDialog.Accepted:
        dicomFilePaths = fileDialog.selectedFiles()
    else:
        slicer.util.errorDisplay("No files selected. Please select DICOM files.")
        return

    if len(dicomFilePaths) == 0:
        slicer.util.errorDisplay("No files selected. Please select DICOM files.")
        return

    try:
        if len(dicomFilePaths) == 1:
            print("[Debug] Loading single DICOM file.")
            loadedNode = slicer.util.loadVolume(dicomFilePaths[0])
            if loadedNode:
                imageData = loadedNode.GetImageData()
                if imageData:
                    dimensions = imageData.GetDimensions()
                    numberOfSlices = dimensions[2]
                    print(f"[Debug] Volume dimensions: {dimensions}. Number of slices: {numberOfSlices}")

                    if numberOfSlices > 1:
                        print("[Info] Multiple slices detected. Prompting user for selection.")
                        selectedSlice = selectSliceToImport(loadedNode)
                        if selectedSlice is not None:
                            loadedNode = reduceVolumeToSingleSlice(loadedNode, selectedSlice)
                            if loadedNode:
                                print(f"[Debug] Successfully reduced volume to slice {selectedSlice}.")
                            else:
                                slicer.util.errorDisplay("Failed to extract the selected slice.")
                                return
                        else:
                            print("[Debug] No slice selected. Proceeding with entire volume.")
                    else:
                        print("[Debug] Single slice detected. Proceeding with loaded volume.")

                reset2DAnd3DViews()
                applyXRayTransform()
                slicer.util.infoDisplay(
                    f"Successfully imported and processed single DICOM file: {dicomFilePaths[0]}"
                )
            else:
                slicer.util.errorDisplay(f"Failed to import file: {dicomFilePaths[0]}")
        else:
            print("[Debug] Loading multiple DICOM files.")
            slicer.dicomDatabase.importFiles(dicomFilePaths)
            slicer.util.infoDisplay(f"Imported {len(dicomFilePaths)} DICOM files to database.")
            slicer.util.selectModule("DICOM")

    except Exception as e:
        slicer.util.errorDisplay(f"Error importing DICOM data: {e}")
        print(f"[Error] Exception during DICOM import: {e}")


def selectSliceToImport(volumeNode):
    """
    Display a dialog to allow the user to select which slice (Z-dimension) to import.
    
    Parameters:
        volumeNode (vtkMRMLScalarVolumeNode): The volume node to inspect.

    Returns:
        int: Selected Z-index to import, or None if canceled.
    """
    # Sprawdź poprawność węzła
    if not isinstance(volumeNode, slicer.vtkMRMLScalarVolumeNode):
        slicer.util.errorDisplay("Provided node is not a valid Scalar Volume Node.")
        return None

    # Pobierz dane obrazu
    imageData = volumeNode.GetImageData()
    if not imageData:
        slicer.util.errorDisplay("No image data found in the volume.")
        return None

    # Pobierz rozszerzenie obrazu
    extent = imageData.GetExtent()
    zMin, zMax = extent[4], extent[5]

    # Debug: Informacja o rozszerzeniu
    print(f"[Debug] Volume extent Z: {zMin} to {zMax}")

    # Jeśli nie ma wielu slice'ów
    if zMax <= zMin:
        slicer.util.infoDisplay("The volume contains only a single slice. No selection needed.")
        return zMin  # Zwróć jedyny slice

    # Generuj listę opcji dla dialogu
    sliceOptions = [f"Slice {z}" for z in range(zMin, zMax + 1)]

    # Debug: Sprawdzenie opcji slice'ów
    print(f"[Debug] Slice options available: {sliceOptions}")

    # Utwórz okno dialogowe
    dialog = qt.QInputDialog()
    dialog.setComboBoxItems(sliceOptions)
    dialog.setWindowTitle("Select Slice to Import")
    dialog.setLabelText("Select the slice to import:")
    dialog.setComboBoxEditable(False)

    # Wyświetl dialog i pobierz wybór
    if dialog.exec_() == qt.QDialog.Accepted:
        selectedSliceText = dialog.textValue()
        try:
            selectedIndex = int(selectedSliceText.split()[1])  # Wyciągnij indeks slice'a
            print(f"[Debug] User selected slice Z={selectedIndex} to import.")
            return selectedIndex
        except ValueError:
            slicer.util.errorDisplay("Invalid slice selection. Please try again.")
            print("[Debug] Invalid slice selection format.")
            return None

    # Anulowano wybór
    print("[Debug] Slice selection canceled.")
    return None

def reduceVolumeToSingleSlice(volumeNode, sliceIndex):
    """
    Extract a specific slice from the volume with proper orientation and centered rotation.

    Parameters:
        volumeNode (vtkMRMLScalarVolumeNode): The input volume node.
        sliceIndex (int): The Z-index of the slice to extract.

    Returns:
        vtkMRMLScalarVolumeNode: A new volume node with the selected slice.
    """
    imageData = volumeNode.GetImageData()
    if not imageData:
        slicer.util.errorDisplay("No image data found in the volume.")
        return None

    extent = imageData.GetExtent()
    if sliceIndex < extent[4] or sliceIndex > extent[5]:
        slicer.util.errorDisplay(f"Invalid slice index: {sliceIndex}")
        return None

    # Create the reslice object
    reslice = vtk.vtkImageReslice()
    reslice.SetInputData(imageData)

    # Center of the slice (in voxel coordinates)
    centerX = (extent[0] + extent[1]) / 2.0
    centerY = (extent[2] + extent[3]) / 2.0
    centerZ = sliceIndex

    # Create the transformation matrix for translation to the center
    translationToCenter = vtk.vtkMatrix4x4()
    translationToCenter.Identity()
    translationToCenter.SetElement(0, 3, -centerX)
    translationToCenter.SetElement(1, 3, -centerY)
    translationToCenter.SetElement(2, 3, -centerZ)

    # Create the rotation matrix (e.g., flip Y and Z axes)
    rotationMatrix = vtk.vtkMatrix4x4()
    rotationMatrix.Identity()
    rotationMatrix.SetElement(0, 0, -1)  
    rotationMatrix.SetElement(1, 1, -1)  # Flip Y-axis
    rotationMatrix.SetElement(2, 2, -1)  # Flip Z-axis

    # Create the transformation matrix for translation back
    translationBack = vtk.vtkMatrix4x4()
    translationBack.Identity()
    translationBack.SetElement(0, 3, centerX)
    translationBack.SetElement(1, 3, centerY)
    translationBack.SetElement(2, 3, centerZ)

    # Combine transformations: TranslationToCenter -> Rotation -> TranslationBack
    combinedMatrix1 = vtk.vtkMatrix4x4()
    vtk.vtkMatrix4x4.Multiply4x4(rotationMatrix, translationToCenter, combinedMatrix1)
    combinedMatrix = vtk.vtkMatrix4x4()
    vtk.vtkMatrix4x4.Multiply4x4(translationBack, combinedMatrix1, combinedMatrix)

    # Apply the transformation matrix to the reslice
    reslice.SetResliceAxes(combinedMatrix)

    # Configure reslice parameters
    reslice.SetOutputExtent(extent[0], extent[1], extent[2], extent[3], sliceIndex, sliceIndex)
    reslice.SetOutputDimensionality(2)  # Extract a 2D slice
    reslice.SetInterpolationModeToNearestNeighbor()  # Avoid artifacts
    reslice.Update()

    # Get the resliced image data
    sliceImageData = reslice.GetOutput()

    # Create a new volume node
    newVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "SelectedSliceVolume")
    newVolumeNode.SetAndObserveImageData(sliceImageData)

    # Copy the origin and spacing
    origin = list(volumeNode.GetOrigin())
    origin[2] += sliceIndex * volumeNode.GetSpacing()[2]  # Adjust Z origin to match slice position
    newVolumeNode.SetOrigin(origin)

    spacing = list(volumeNode.GetSpacing())
    spacing[2] = 1.0  # Set Z-spacing to 1 for a single slice
    newVolumeNode.SetSpacing(spacing)
    newVolumeNode.SetName(volumeNode.GetName() + "_Slice" + str(sliceIndex))

    # Copy display properties
    originalDisplayNode = volumeNode.GetDisplayNode()
    if originalDisplayNode:
        newDisplayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeDisplayNode")
        newDisplayNode.Copy(originalDisplayNode)
        newVolumeNode.SetAndObserveDisplayNodeID(newDisplayNode.GetID())

    # Remove the original volume node
    slicer.mrmlScene.RemoveNode(volumeNode)
    print("[Debug] Original volume node removed from the scene.")

    return newVolumeNode

def rotateGreenView():   
    """
    Rotate the image in the green (axial) view by 90 degrees clockwise around the X-axis.
    """
    greenSliceWidget = slicer.app.layoutManager().sliceWidget("Green")
    greenSliceLogic = greenSliceWidget.sliceLogic()
    greenSliceNode = greenSliceLogic.GetSliceNode()

    # Get the current SliceToRAS matrix
    sliceToRAS = greenSliceNode.GetSliceToRAS()  # No arguments are passed

    # Create a rotation matrix for 90 degrees around the X-axis
    rotationMatrix = vtk.vtkMatrix4x4()
    rotationMatrix.Identity()

    angle = 90  # Degrees
    cosTheta = math.cos(math.radians(angle))
    sinTheta = math.sin(math.radians(angle))

    # Set elements for rotation around the X-axis
    rotationMatrix.SetElement(0, 0, cosTheta)
    rotationMatrix.SetElement(0, 2, -sinTheta)
    rotationMatrix.SetElement(2, 0, sinTheta)
    rotationMatrix.SetElement(2, 2, cosTheta)

    # Combine the current matrix with the rotation
    newSliceToRAS = vtk.vtkMatrix4x4()
    vtk.vtkMatrix4x4.Multiply4x4(rotationMatrix, sliceToRAS, newSliceToRAS)

    # Set the new SliceToRAS matrix
    greenSliceNode.GetSliceToRAS().DeepCopy(newSliceToRAS)
    greenSliceLogic.FitSliceToAll()
    slicer.app.processEvents()  # Update GUI    


def importCTFolder():
    """
    Import a folder containing DICOM files for a CT scan into 3D Slicer.
    Automatically loads the most detailed series (highest number of images),
    and resets the scene and variables. Disables the scale SpinBox.
    """
    # Reset the scene and global variables
    resetSceneAndVariables()

    try:
        # Open file dialog to select a folder
        folderDialog = qt.QFileDialog()
        folderDialog.setFileMode(qt.QFileDialog.Directory)
        folderDialog.setWindowTitle("Select Folder Containing CT DICOM Files")

        if folderDialog.exec_() == qt.QDialog.Accepted:
            selectedFolder = folderDialog.selectedFiles()[0]
        else:
            slicer.util.errorDisplay("No folder selected. Please select a folder containing DICOM files.")
            return

        if not os.path.exists(selectedFolder):
            slicer.util.errorDisplay(f"Selected folder does not exist: {selectedFolder}")
            return

        print(f"[Debug] Importing DICOM folder: {selectedFolder}")

        # Initialize DICOM database
        with DICOMUtils.TemporaryDICOMDatabase() as tempDatabase:
            # Import the directory into the temporary database
            DICOMUtils.importDicom(selectedFolder, tempDatabase)

            # Check number of patients in the database
            patients = tempDatabase.patients()
            if not patients:
                slicer.util.errorDisplay(
                    "No valid DICOM patients found in the selected folder. "
                    "Please ensure the folder contains valid DICOM files."
                )
                print("[Debug] No patients found in the imported DICOM folder.")
                return
            print(f"[Debug] Found {len(patients)} patients in the DICOM database.")

            # Get the first patient's studies
            patientUID = patients[0]
            studies = tempDatabase.studiesForPatient(patientUID)
            if not studies:
                slicer.util.errorDisplay(f"No studies found for patient UID: {patientUID}")
                print(f"[Debug] No studies found for patient UID: {patientUID}")
                return
            print(f"[Debug] Found {len(studies)} studies for patient UID: {patientUID}")

            # Get the first study's series and sort by number of images
            studyUID = studies[0]
            series = tempDatabase.seriesForStudy(studyUID)
            if not series:
                slicer.util.errorDisplay(f"No series found for study UID: {studyUID}")
                print(f"[Debug] No series found for study UID: {studyUID}")
                return

            # Sort series by the number of images in descending order
            sortedSeries = sorted(
                series,
                key=lambda s: len(tempDatabase.filesForSeries(s)),
                reverse=True
            )

            print(f"[Debug] Found {len(sortedSeries)} series for study UID: {studyUID}.")
            print("[Debug] Series sorted by number of images (descending):")
            for s in sortedSeries:
                print(f"Series UID: {s}, Number of Images: {len(tempDatabase.filesForSeries(s))}")

            # Load the most detailed series (first in sorted list)
            seriesUID = sortedSeries[0]
            try:
                DICOMUtils.loadSeriesByUID([seriesUID])  # Pass as a list
                print(f"[Debug] Successfully loaded most detailed series UID: {seriesUID}")
            except Exception as loadError:
                slicer.util.errorDisplay(f"Error loading series UID: {seriesUID}\n{loadError}")
                print(f"[Debug] Error loading series UID: {seriesUID}\n{loadError}")
                return

        slicer.util.infoDisplay(f"Successfully imported and loaded DICOM files from folder: {selectedFolder}")

        # Disable and reset the SpinBox
        updateSpinBoxForSeries()

        # Switch to Four-Up view
        layoutManager = slicer.app.layoutManager()
        if layoutManager:
            layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
            print("[Debug] Switched to Four-Up view.")

    except Exception as e:
        slicer.util.errorDisplay(f"Error importing CT folder: {e}")
        print(f"[Debug] Error during CT folder import: {e}")

def updateSpinBoxForSeries():
    """
    Disable the SpinBox and reset its value to 100 for series.
    """
    global xrayScaleSpinBox
    if xrayScaleSpinBox:
        xrayScaleSpinBox.blockSignals(True)  # Prevent signal firing
        xrayScaleSpinBox.setValue(100)
        xrayScaleSpinBox.setEnabled(False)  # Disable the SpinBox
        xrayScaleSpinBox.blockSignals(False)
        print("[Debug] SpinBox disabled and reset to 100 for series import.")

def enableSpinBoxForXRay():
    """
    Enable the SpinBox after a single DICOM file is loaded and X-ray transform is applied.
    """
    global xrayScaleSpinBox
    if xrayScaleSpinBox:
        xrayScaleSpinBox.setEnabled(True)
        print("[Debug] SpinBox enabled for X-ray scaling.")

def resetSceneAndVariables():
    """
    Reset the 3D Slicer scene and clear all global variables.
    """
    global L_Cup_Node, R_Cup_Node, L_Tr_Node, R_Tr_Node, Head_Node
    global L_Cup_Transform, R_Cup_Transform, L_Tr_Transform, R_Tr_Transform
    global Head_Transform_L, Head_Transform_R, Head_Visible, distanceLabel
    global currentSegmentationNode, currentSegmentID, originalPinSize, currentPinSize, maxPinSize
    global Ischial_Line_Node, XRay_Transform_Node, XRay_Mode_Enabled, xrayScaleSpinBox

    print("[Debug] Resetting scene and global variables.")

    # Reset global variables
    L_Cup_Node = None
    R_Cup_Node = None
    L_Tr_Node = None
    R_Tr_Node = None
    Head_Node = None

    L_Cup_Transform = None
    R_Cup_Transform = None
    L_Tr_Transform = None
    R_Tr_Transform = None

    Head_Transform_L = None
    Head_Transform_R = None
    Head_Visible = False

    distanceLabel = None

    currentSegmentationNode = None
    currentSegmentID = None
    originalPinSize = None
    currentPinSize = None
    maxPinSize = None

    Ischial_Line_Node = None
    XRay_Transform_Node = None
    XRay_Mode_Enabled = False

    # Clear the scene
    slicer.mrmlScene.Clear(0)

    # Reset the SpinBox state
    updateSpinBoxForSeries()

    print("[Debug] Scene cleared and SpinBox reset.")

def addOrUpdateDynamicSegmentLabel(segmentationNode, segmentID, labelText, textOffset=-5, textScale=3.0, textColor=(1, 1, 1)):
    """
    Create or update a label for a segment, ensuring it appears in the 3D and 2D views,
    and follows the segment's transform. Removes all existing label nodes before creating a new one.

    Parameters:
        segmentationNode (vtkMRMLSegmentationNode): The segmentation node containing the segment.
        segmentID (str): The ID of the segment.
        labelText (str): The text to display in the label.
        textOffset (float): Z-axis offset for the label's position.
        textScale (float): Scale of the label's text.
        textColor (tuple): RGB color for the label (default is white).
    """
    if not segmentationNode or not segmentID:
        print(f"[Label Error] Invalid segmentation node or segment ID for '{labelText}'.")
        return

    # Usunięcie wszystkich istniejących węzłów etykiet związanych z tym segmentem
    allNodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLMarkupsFiducialNode")
    for i in range(allNodes.GetNumberOfItems()):
        fiducialNode = allNodes.GetItemAsObject(i)
        if fiducialNode.GetName() and fiducialNode.GetName().startswith(segmentID):
            slicer.mrmlScene.RemoveNode(fiducialNode)
            print(f"[Debug] Removed previous label node: {fiducialNode.GetName()}")

    # Get the global center of the segment
    globalCenter = getSegmentCenter(segmentationNode, segmentID)
    if globalCenter == [0.0, 0.0, 0.0]:
        print(f"[Label Error] Cannot calculate center for segment '{segmentID}'.")
        return
    print(f"[Debug] Global center for segment '{segmentID}': {globalCenter}")

    # Adjust position for the label (Z-offset)
    globalLabelPosition = [
        globalCenter[0],
        globalCenter[1],
        globalCenter[2] + textOffset
    ]

    # Transform to local coordinates if the segmentation has a transform
    transformNode = segmentationNode.GetParentTransformNode()
    if transformNode:
        transformMatrix = vtk.vtkMatrix4x4()
        transformNode.GetMatrixTransformToWorld(transformMatrix)
        # Invert the transform to convert global to local
        invertedMatrix = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(transformMatrix, invertedMatrix)

        # Apply the inverted transform to calculate local label position
        localLabelPosition = [0.0, 0.0, 0.0, 1.0]
        globalLabelPosition.append(1.0)  # Add homogeneous coordinate
        invertedMatrix.MultiplyPoint(globalLabelPosition, localLabelPosition)
        localLabelPosition = localLabelPosition[:3]  # Remove the homogeneous component
    else:
        # No transform, local position is the same as global
        localLabelPosition = globalLabelPosition

    print(f"[Debug] Local label position for '{labelText}': {localLabelPosition}")

    # Create a new fiducial node for the label
    fiducialNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", f"{segmentID}_{labelText}")
    fiducialNode.AddControlPoint(localLabelPosition)
    print(f"[Debug] Created label node '{fiducialNode.GetName()}' at position: {localLabelPosition}")

    # Customize display properties
    displayNode = fiducialNode.GetDisplayNode()
    if not displayNode:
        fiducialNode.CreateDefaultDisplayNodes()
        displayNode = fiducialNode.GetDisplayNode()

    if displayNode:
        displayNode.SetGlyphScale(0)  # Hide the marker glyph
        displayNode.SetTextScale(textScale)  # Adjust text scale
        displayNode.SetColor(*textColor)  # Set text color
        displayNode.SetVisibility(True)  # Ensure visibility in both 2D and 3D
        displayNode.SetVisibility2D(True)  # Ensure visibility in 2D views
        displayNode.SetVisibility3D(True)  # Ensure visibility in 3D views
        print(f"[Debug] Updated display properties for '{fiducialNode.GetName()}' (Scale={textScale}, Color={textColor})")
    else:
        print(f"[Label Error] Could not configure display properties for '{fiducialNode.GetName()}'.")

    # Ensure the fiducial follows the transform of the segmentation
    if transformNode:
        fiducialNode.SetAndObserveTransformNodeID(transformNode.GetID())
        print(f"[Debug] Label '{fiducialNode.GetName()}' linked to transform: {transformNode.GetName()}")


def importSTLAsSegmentation(side="Right", updateExisting=False):
    """
    Import an STL file as a segmentation and immediately show the transform manipulator.
    Automatically sets a label based on the size mapping or default size.
    Adjusts position to be 5 cm from the center in the X-axis, regardless of other reference points.
    """
    import csv

    global currentSegmentationNode, currentSegmentID, originalPinSize, currentPinSize, maxPinSize, sizeMapping

    if updateExisting:
        if not currentSegmentationNode:
            print("[Info] No segmentation found to update.")
            return

        # Update the existing segmentation transform
        transformNodeID = currentSegmentationNode.GetTransformNodeID()
        if not transformNodeID:
            print("[Info] No transform node found for the current segmentation.")
            return

        transformNode = slicer.mrmlScene.GetNodeByID(transformNodeID)
        if not transformNode:
            print("[Info] Transform node not found.")
            return

        # Reset the current transform to ensure consistent behavior
        vtk_transform = vtk.vtkTransform()

        # Default to volume center if no target transform is found
        volumeNode = slicer.util.getNode("vtkMRMLScalarVolumeNode*")
        bounds = [0.0] * 6
        volumeNode.GetRASBounds(bounds)
        volumeCenter = [
            (bounds[0] + bounds[1]) / 2.0,
            (bounds[2] + bounds[3]) / 2.0,
            (bounds[4] + bounds[5]) / 2.0,
        ]
        vtk_transform.Translate(volumeCenter)

        # Adjust position 5 cm from the center in the X-axis and 5 cm down in the Z-axis
        vtk_transform.Translate(50.0 if side == "Right" else -50.0, 0, -50.0)

        # Apply mirroring for "Left" along the X-axis
        if side == "Left":
            vtk_transform.Scale(-1.0, 1.0, 1.0)  # Mirror along X-axis

        # Set the new transform matrix
        newMatrix = vtk.vtkMatrix4x4()
        vtk_transform.GetMatrix(newMatrix)
        transformNode.SetMatrixTransformToParent(newMatrix)

        print(f"Segmentation updated to side: {side}.")
        return

    # Open file dialog to select an STL file
    fileDialog = qt.QFileDialog()
    fileDialog.setFileMode(qt.QFileDialog.ExistingFile)
    fileDialog.setNameFilter("STL Files (*.stl)")
    fileDialog.setWindowTitle("Import STL File")

    if fileDialog.exec_() == qt.QDialog.Accepted:
        filePath = fileDialog.selectedFiles()[0]
    else:
        slicer.util.errorDisplay("No file selected. Please select an STL file.")
        return

    # Check if file exists
    if not os.path.exists(filePath):
        slicer.util.errorDisplay(f"File '{filePath}' does not exist.")
        return

    # Attempt to load the size mapping from a CSV file
    csvFilePath = filePath.replace(".stl", ".csv")
    sizeMapping = {}

    if os.path.exists(csvFilePath):
        try:
            with open(csvFilePath, newline='') as csvfile:
                reader = csv.DictReader(csvfile)  # Automatyczne obsłużenie nagłówków
                for row in reader:
                    size_name = float(row["Size Name"])
                    value = float(row["Value"])
                    axis = row["Axis"]
                    sizeMapping[size_name] = {"value": value, "axis": axis}
            print(f"[Debug] Size mapping loaded from '{csvFilePath}': {sizeMapping}")
        except Exception as e:
            slicer.util.errorDisplay(f"Error loading size mapping from '{csvFilePath}': {e}")
            sizeMapping = None
    else:
        slicer.util.infoDisplay(
            "No size mapping file found for the imported object. Scaling will proceed in 1 mm increments vertically, with proportional adjustments to other dimensions."
        )
        sizeMapping = None

    # Read STL data using vtkSTLReader
    stlReader = vtk.vtkSTLReader()
    stlReader.SetFileName(filePath)
    stlReader.Update()

    polyData = stlReader.GetOutput()
    if not polyData or polyData.GetNumberOfPoints() == 0:
        slicer.util.errorDisplay(f"Failed to read STL file '{filePath}'.")
        return

    # Create a new segmentation node
    segmentationNode = slicer.mrmlScene.AddNewNodeByClass(
        "vtkMRMLSegmentationNode",
        os.path.basename(filePath).replace(".stl", f" {side} Segmentation")
    )

    # Visualization settings
    segmentationDisplayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationDisplayNode")
    segmentationNode.SetAndObserveDisplayNodeID(segmentationDisplayNode.GetID())
    segmentationDisplayNode.SetVisibility3D(True)
    segmentationDisplayNode.SetVisibility2DFill(True)
    segmentationDisplayNode.SetVisibility2DOutline(True)

    # Add segment based on STL
    segment = slicer.vtkSegment()
    segment.SetName(os.path.basename(filePath).replace(".stl", ""))
    segment.AddRepresentation(
        slicer.vtkSegmentationConverter.GetClosedSurfaceRepresentationName(),
        polyData
    )
    segmentationNode.GetSegmentation().AddSegment(segment)

    # Create transform node and assign it to the segmentation
    transformNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode", f"{segment.GetName()} Transform")
    segmentationNode.SetAndObserveTransformNodeID(transformNode.GetID())

    # Default position: center of the volume
    volumeNode = slicer.util.getNode("vtkMRMLScalarVolumeNode*")
    bounds = [0.0] * 6
    volumeNode.GetRASBounds(bounds)
    volumeCenter = [
        (bounds[0] + bounds[1]) / 2.0,
        (bounds[2] + bounds[3]) / 2.0,
        (bounds[4] + bounds[5]) / 2.0,
    ]
    vtk_transform = vtk.vtkTransform()
    vtk_transform.Translate(volumeCenter)

    # Adjust position 5 cm from the center in the X-axis and 5 cm down in the Z-axis
    vtk_transform.Translate(50.0 if side == "Right" else -50.0, 0, -50.0)

    # Apply mirroring for "Left" along the X-axis
    if side == "Left":
        vtk_transform.Scale(-1.0, 1.0, 1.0)  # Mirror along X-axis

    newMatrix = vtk.vtkMatrix4x4()
    vtk_transform.GetMatrix(newMatrix)
    transformNode.SetMatrixTransformToParent(newMatrix)

    # Initialize global variables for scaling
    currentSegmentationNode = segmentationNode
    currentSegmentID = segment.GetName()
    bounds = [0.0] * 6
    polyData.GetBounds(bounds)
    originalPinSize = bounds[1] - bounds[0]  # Initial size along X-axis
    currentPinSize = originalPinSize
    maxPinSize = originalPinSize + 19  # Allow growth up to 19 mm larger

    # Set initial label based on size mapping or default size
    if sizeMapping:
        sizeNames = list(sizeMapping.keys())
        initialSizeName = sizeNames[0]
    else:
        initialSizeName = f"{currentPinSize:.1f}"

    addOrUpdateDynamicSegmentLabel(
        segmentationNode=currentSegmentationNode,
        segmentID=currentSegmentID,
        labelText=str(initialSizeName),  # Tekst etykiety
        textOffset=-5,
        textScale=3.0,
        textColor=(1.0, 1.0, 1.0)  # Biały kolor
    )

    print(f"Label initialized to: {initialSizeName}")

    # Enable manipulator for the transform node
    transformNode.CreateDefaultDisplayNodes()
    displayNode = transformNode.GetDisplayNode()
    if displayNode:
        displayNode.SetEditorVisibility(True)  # Show manipulators
        print(f"[Debug] Manipulator enabled for transform: {transformNode.GetName()}")

    print(f"Imported STL file '{filePath}' as {side} segmentation. Original size: {originalPinSize:.2f} mm.")

    # Apply mirroring for "Left" along the X-axis
    if side == "Left":
        vtk_transform.Scale(-1.0, 1.0, 1.0)  # Mirror along X-axis

    newMatrix = vtk.vtkMatrix4x4()
    vtk_transform.GetMatrix(newMatrix)
    transformNode.SetMatrixTransformToParent(newMatrix)

    # Initialize global variables for scaling
    currentSegmentationNode = segmentationNode
    currentSegmentID = segment.GetName()
    bounds = [0.0] * 6
    polyData.GetBounds(bounds)
    originalPinSize = bounds[1] - bounds[0]  # Initial size along X-axis
    currentPinSize = originalPinSize
    maxPinSize = originalPinSize + 19  # Allow growth up to 19 mm larger

    # Set initial label based on size mapping or default size
    if sizeMapping:
        sizeNames = list(sizeMapping.keys())
        initialSizeName = sizeNames[0]
    else:
        initialSizeName = f"{currentPinSize:.1f}"

    addOrUpdateDynamicSegmentLabel(
        segmentationNode=currentSegmentationNode,
        segmentID=currentSegmentID,
        labelText=str(initialSizeName),  # Tekst etykiety
        textOffset=-5,
        textScale=3.0,
        textColor=(1.0, 1.0, 1.0)  # Biały kolor
    )

    print(f"Label initialized to: {initialSizeName}")

    # Enable manipulator for the transform node
    transformNode.CreateDefaultDisplayNodes()
    displayNode = transformNode.GetDisplayNode()
    if displayNode:
        displayNode.SetEditorVisibility(True)  # Show manipulators
        print(f"[Debug] Manipulator enabled for transform: {transformNode.GetName()}")

    print(f"Imported STL file '{filePath}' as {side} segmentation. Original size: {originalPinSize:.2f} mm.")

def applyXRayTransform():
    """
    Toggle X-ray mode:
    - First press: Ensure Coronal visibility, maximize Coronal view (Green slice), reset views.
    - Second press: Undo transformation, restore Four-Up layout, reset views, and lock scale spinbox.
    """
    global XRay_Mode_Enabled, XRay_Transform_Node, originalLayout

    layoutManager = slicer.app.layoutManager()

    if XRay_Mode_Enabled:
        # Second press: Undo X-ray transform and restore Four-Up layout
        print("[Debug] Undoing X-ray transform.")

        if XRay_Transform_Node:
            try:
                volumeNode = slicer.util.getNode("vtkMRMLScalarVolumeNode*")
                if volumeNode:
                    # Remove the transform from the volume
                    volumeNode.SetAndObserveTransformNodeID(None)
                    print("[Debug] Transform removed from volume.")
            except slicer.util.MRMLNodeNotFoundException:
                print("[Debug] No volume node found to undo transformation.")

            # Delete the transform node after detaching it
            slicer.mrmlScene.RemoveNode(XRay_Transform_Node)
            XRay_Transform_Node = None
            print("[Debug] X-ray transform node deleted.")

        # Restore the Four-Up layout and reset views
        layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
        reset2DAnd3DViews()

        # Lock the scale spinbox and reset its value to 100
        xrayScaleSpinBox.blockSignals(True)  # Prevent unwanted signal handling during reset
        xrayScaleSpinBox.setEnabled(False)
        xrayScaleSpinBox.setValue(100)
        xrayScaleSpinBox.blockSignals(False)

        XRay_Mode_Enabled = False  # Toggle mode off
        print("[Debug] X-ray mode disabled. Transformations undone.")

    else:
        # First press: Apply X-ray transform and maximize coronal view
        print("[Debug] Applying X-ray transform.")
        try:
            volumeNode = slicer.util.getNode("vtkMRMLScalarVolumeNode*")
        except slicer.util.MRMLNodeNotFoundException:
            slicer.util.errorDisplay("No volume loaded. Please load a volume to enable X-ray mode.")
            return

        # Save the current layout
        if originalLayout is None:
            originalLayout = layoutManager.layout

        # Apply transformation for Coronal visibility
        if not XRay_Transform_Node:
            XRay_Transform_Node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode", "X-Ray Transform")

        vtk_transform = vtk.vtkTransform()
        vtk_transform.Identity()  # Start with identity matrix
        vtk_transform.RotateX(90)  # Rotate to align to Coronal (fix orientation)
        vtk_transform.Scale(1/1.15, 1/1.15, 1/1.15)


        transformMatrix = vtk.vtkMatrix4x4()
        vtk_transform.GetMatrix(transformMatrix)
        XRay_Transform_Node.SetMatrixTransformToParent(transformMatrix)
        volumeNode.SetAndObserveTransformNodeID(XRay_Transform_Node.GetID())
        print("[Debug] Transformation applied to align to Coronal.")

        # Center the slice views and set background
        for color in ["Red", "Green", "Yellow"]:
            sliceCompositeNode = slicer.mrmlScene.GetNodeByID(f"vtkMRMLSliceCompositeNode{color}")
            sliceCompositeNode.SetBackgroundVolumeID(volumeNode.GetID())  # Set background image
            sliceLogic = slicer.app.layoutManager().sliceWidget(color).sliceLogic()
            sliceLogic.FitSliceToAll()  # Ensure slice fits data
            sliceLogic.SnapSliceOffsetToIJK()  # Center the image

        # Maximize green slice view
        layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpGreenSliceView)

        # Unlock the scale spinbox and set its value to 115
        xrayScaleSpinBox.blockSignals(True)  # Prevent unwanted signal handling during update
        xrayScaleSpinBox.setEnabled(True)
        xrayScaleSpinBox.setValue(115)
        xrayScaleSpinBox.blockSignals(False)

        XRay_Mode_Enabled = True  # Toggle mode on
        print("[Debug] X-ray mode enabled with proper Coronal visibility.")
        
def getCurrentXRayScale():
    """
    Get the current scaling factor from the X-Ray Transform, converted to percentage.
    :return: Scale percentage (e.g., 115 for 115% scaling).
    """
    global XRay_Transform_Node

    if not XRay_Transform_Node:
        return 115  # Default to 115% if transform does not exist

    transformMatrix = vtk.vtkMatrix4x4()
    XRay_Transform_Node.GetMatrixTransformToWorld(transformMatrix)

    # Assume uniform scaling, read scale factor from X-axis
    scaleFactor = transformMatrix.GetElement(0, 0)
    scalePercent = int(scaleFactor * 100)  # Convert to percentage
    return scalePercent    
    
def getCurrentXRayScale():
    """
    Get the current scaling factor from the X-Ray Transform, converted to percentage.
    :return: Scale percentage (e.g., 115 for 115% scaling).
    """
    global XRay_Transform_Node

    if not XRay_Transform_Node:
        return 115  # Default to 115% if transform does not exist

    transformMatrix = vtk.vtkMatrix4x4()
    XRay_Transform_Node.GetMatrixTransformToWorld(transformMatrix)

    # Assume uniform scaling, read scale factor from X-axis
    scaleFactor = transformMatrix.GetElement(0, 0)
    scalePercent = round(scaleFactor * 100)  # Convert to percentage using standard rounding
    return scalePercent

        
# Globalna zmienna do śledzenia bieżącej skali
currentXRayScale = 115  # Domyślna wartość po imporcie

# Funkcja zdefiniowana na poziomie globalnym
def getLineMeasurement():
    """
    Get the measurement of the line named 'xray marker line'.
    """
    try:
        lineNode = slicer.mrmlScene.GetFirstNodeByName("xray marker line")
        if lineNode:
            startPoint = [0.0, 0.0, 0.0]
            endPoint = [0.0, 0.0, 0.0]
            lineNode.GetNthControlPointPositionWorld(0, startPoint)
            lineNode.GetNthControlPointPositionWorld(1, endPoint)
            length = vtk.vtkMath.Distance2BetweenPoints(startPoint, endPoint) ** 0.5
            return length
        else:
            slicer.util.errorDisplay("Line 'xray marker line' not found.")
            return None
    except Exception as e:
        slicer.util.errorDisplay(f"Failed to get line measurement: {e}")
        return None

def applyMagnification():
    """
    Calculate the magnification scale directly from measured vs actual size 
    and update the X-ray scale spinbox if the result is within a valid range.
    """
    try:
        print("[Debug] Starting applyMagnification...")

        # Pobranie wartości z QComboBox
        currentIndex = actualSizeInput.currentIndex
        if currentIndex < 0:
            slicer.util.errorDisplay("Please select a valid size from the dropdown.")
            return

        actualSize = actualSizeInput.itemData(currentIndex)
        if actualSize is None:
            slicer.util.errorDisplay("No valid data found for the selected size.")
            return

        actualSize = float(actualSize)  # Konwersja na float
        print(f"[Debug] actualSize from combo box: {actualSize}")

        # Pobranie wartości pomiaru linii
        lineMeasurement = getLineMeasurement()
        if lineMeasurement is None:
            return

        # Wstawienie wartości pomiaru linii do pola tekstowego
        measuredSizeInput.setText(f"{lineMeasurement:.2f}")

        # Pobranie wartości z pola tekstowego dla measured size
        measuredSizeText = measuredSizeInput.text.strip()
        if not measuredSizeText:
            slicer.util.errorDisplay("Measured size cannot be empty.")
            return

        measuredSize = float(measuredSizeText)
        print(f"[Debug] measuredSize: {measuredSize}")

        if actualSize <= 0:
            slicer.util.errorDisplay("Actual size must be greater than zero.")
            return

        # Oblicz bezpośrednie powiększenie (bez korekty o aktualną skalę)
        magnification = (measuredSize / actualSize) * 100
        print(f"[Debug] Calculated magnification: {magnification}")

        # Matematyczne zaokrąglanie
        roundedMagnification = round(magnification)
        print(f"[Debug] Rounded magnification: {roundedMagnification}")

        # Sprawdzenie zakresu
        if roundedMagnification < 100 or roundedMagnification > 125:
            slicer.util.errorDisplay(
                "Calculated magnification is out of the valid range (100-125%). "
                "Please verify the input values for accuracy."
            )
            print("[Warning] Magnification out of range. Not applying to spinbox.")
            return

        # Ustawienie wartości w spinboksie
        xrayScaleSpinBox.setValue(roundedMagnification)
        print(f"[Debug] Spinbox value set to: {xrayScaleSpinBox.value()}")

    except ValueError as e:
        slicer.util.errorDisplay(f"Invalid input: {e}")
    except Exception as e:
        print(f"[Error] Unexpected error in applyMagnification: {e}")


def resetScene():
    """
    Reset the scene by removing all segmentations, transformations, and labels,
    except for the X-ray transform (if applied).
    Also resets spinboxes to their default values and clears checkbox states.
    """
    global L_Cup_Node, R_Cup_Node, L_Tr_Node, R_Tr_Node
    global L_Cup_Transform, R_Cup_Transform, L_Tr_Transform, R_Tr_Transform
    global Head_Transform_L, Head_Transform_R, Ischial_Line_Node, XRay_Transform_Node
    global cupSizeSpinBoxL, cupSizeSpinBoxR  # Spinboxes
    global L_Tr_Checkbox, R_Tr_Checkbox, L_Cup_Checkbox, R_Cup_Checkbox, Ischial_Checkbox  # Checkboxes

    try:
        print("[Debug] Resetting scene by removing segmentations, transformations, and labels.")

        # Reset all checkboxes before removing nodes
        checkboxes = [L_Tr_Checkbox, R_Tr_Checkbox, L_Cup_Checkbox, R_Cup_Checkbox, Ischial_Checkbox]
        for checkbox in checkboxes:
            try:
                if checkbox is not None and callable(getattr(checkbox, "setChecked", None)):
                    checkbox.setChecked(False)  # Uncheck checkbox
                    checkbox.setEnabled(False)  # Disable checkbox
                    print(f"[Debug] Checkbox {checkbox.text() if hasattr(checkbox, 'text') else 'Unknown'} reset.")
                else:
                    print(f"[Warning] Checkbox is None or not callable: {checkbox}")
            except Exception as e:
                print(f"[Error] Failed to reset checkbox: {e}")
                continue  # Ignore the error and move to the next checkbox

        # Save the current X-ray transform matrix if it exists
        xrayTransformMatrix = vtk.vtkMatrix4x4()
        if XRay_Transform_Node:
            XRay_Transform_Node.GetMatrixTransformToParent(xrayTransformMatrix)
            print("[Debug] X-ray transform matrix saved.")

        # Collect nodes to remove
        nodesToRemove = []
        for node in slicer.mrmlScene.GetNodes():
            if isinstance(node, (slicer.vtkMRMLSegmentationNode, slicer.vtkMRMLTransformNode, slicer.vtkMRMLMarkupsNode)):
                if node != XRay_Transform_Node:  # Exclude the X-ray transform
                    nodesToRemove.append(node)

        # Remove collected nodes
        for node in nodesToRemove:
            slicer.mrmlScene.RemoveNode(node)
            print(f"[Debug] Removed node: {node.GetName()}")

        # Reapply the X-ray transform if it was applied
        if XRay_Transform_Node:
            volumeNode = slicer.util.getNode("vtkMRMLScalarVolumeNode*")
            if volumeNode:
                XRay_Transform_Node.SetMatrixTransformToParent(xrayTransformMatrix)
                volumeNode.SetAndObserveTransformNodeID(XRay_Transform_Node.GetID())
                print("[Debug] Reapplied X-ray transformation to the loaded volume.")

        # Reset global variables
        L_Cup_Node = None
        R_Cup_Node = None
        L_Tr_Node = None
        R_Tr_Node = None
        L_Cup_Transform = None
        R_Cup_Transform = None
        L_Tr_Transform = None
        R_Tr_Transform = None
        Head_Transform_L = None
        Head_Transform_R = None
        Ischial_Line_Node = None

        # Reset spinboxes for cup size
        if cupSizeSpinBoxL is not None:
            cupSizeSpinBoxL.blockSignals(True)
            cupSizeSpinBoxL.setValue(50)
            cupSizeSpinBoxL.blockSignals(False)
            print("[Debug] Reset left cup size spinbox to 50.")

        if cupSizeSpinBoxR is not None:
            cupSizeSpinBoxR.blockSignals(True)
            cupSizeSpinBoxR.setValue(50)
            cupSizeSpinBoxR.blockSignals(False)
            print("[Debug] Reset right cup size spinbox to 50.")

        # Refresh views
        reset2DAnd3DViews()
        slicer.app.processEvents()

        print("[Debug] Scene reset completed successfully.")

    except Exception as e:
        slicer.util.errorDisplay(f"Error during scene reset: {e}")
        print(f"[Debug] Error during scene reset: {e}")

    finally:
        print("[Debug] Reset function executed.")


   
def createSphere(diameter):
    """
    Create a sphere with the specified diameter and center it at (0, 0, 0).
    """
    radius = diameter / 2.0  # Convert diameter to radius
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(radius)
    sphere.SetThetaResolution(50)
    sphere.SetPhiResolution(50)
    sphere.SetCenter(0, 0, 0)  # Center the sphere at (0, 0, 0)
    sphere.Update()

    # Debugging the number of points and cells
    output = sphere.GetOutput()
    numPoints = output.GetNumberOfPoints()
    numCells = output.GetNumberOfCells()
    print(f"[Debug] Sphere created with {numPoints} points and {numCells} cells.")

    return output

def createCup(diameter, left=True):
    """
    Create a hemisphere (cup) centered such that its flat surface lies along the X=0 plane,
    and the Z-axis passes through the center of the original sphere.
    """
    radius = diameter / 2.0  # Convert diameter to radius
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(radius)
    sphere.SetThetaResolution(50)
    sphere.SetPhiResolution(50)
    sphere.SetCenter(0, 0, 0)  # Center the sphere at (0, 0, 0)
    sphere.Update()

    # Create a clipping plane
    plane = vtk.vtkPlane()
    plane.SetOrigin(0, 0, 0)  # Clipping plane at the center of the sphere
    plane.SetNormal(-1 if left else 1, 0, 0)  # Normal points along X-axis, depending on side

    # Clip the sphere
    clipper = vtk.vtkClipPolyData()
    clipper.SetInputConnection(sphere.GetOutputPort())
    clipper.SetClipFunction(plane)
    clipper.Update()

    # Center the clipped geometry along the Z-axis
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transform = vtk.vtkTransform()
    transform.Translate(0, 0, 0)  # Ensure center of sphere is aligned with Z-axis
    transformFilter.SetInputConnection(clipper.GetOutputPort())
    transformFilter.SetTransform(transform)
    transformFilter.Update()

    # Debugging information
    output = transformFilter.GetOutput()
    numPoints = output.GetNumberOfPoints()
    numCells = output.GetNumberOfCells()
    print(f"[Debug] Cup created with {numPoints} points and {numCells} cells, aligned along Z-axis.")

    return output

# Globalna deklaracja zmiennej
Ischial_Line_Node = None

def addOrToggleLine(name="Ischial"):
    """
    Create or toggle visibility of a Markups line node in the scene.
    """
    global Ischial_Line_Node  # Upewnij się, że zmienna jest globalna

    # Check if the line node already exists
    if Ischial_Line_Node:
        # Toggle visibility of the line
        displayNode = Ischial_Line_Node.GetDisplayNode()
        if displayNode:
            currentVisibility = displayNode.GetVisibility()
            displayNode.SetVisibility(not currentVisibility)
        return

    # Check if a volume is loaded before creating the line
    if not isVolumeLoaded():
        slicer.util.errorDisplay("No volume loaded. Please load a volume before adding a line.")
        return

    # Get the center of the volume
    try:
        volumeNode = slicer.util.getNode("vtkMRMLScalarVolumeNode*")
        bounds = [0.0] * 6
        volumeNode.GetRASBounds(bounds)
        volumeCenter = [
            (bounds[0] + bounds[1]) / 2.0,  # X-center
            (bounds[2] + bounds[3]) / 2.0,  # Y-center
            (bounds[4] + bounds[5]) / 2.0 - 50.0,  # Z-center, shifted 50 mm down
        ]
    except slicer.util.MRMLNodeNotFoundException:
        slicer.util.errorDisplay("Failed to calculate volume center. Please check the loaded volume.")
        return

    # Create a new MarkupsLine node
    Ischial_Line_Node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", name)

    # Set control points for the line based on the volume center
    startPoint = [volumeCenter[0] - 50, volumeCenter[1], volumeCenter[2]]  # Start point 50 mm to the left
    endPoint = [volumeCenter[0] + 50, volumeCenter[1], volumeCenter[2]]    # End point 50 mm to the right
    Ischial_Line_Node.AddControlPoint(startPoint)
    Ischial_Line_Node.AddControlPoint(endPoint)

    # Customize display properties
    displayNode = Ischial_Line_Node.GetDisplayNode()
    if displayNode:
        displayNode.SetSelectedColor(1, 0, 0)  # Red color for the line
        displayNode.SetLineThickness(0.5)      # Set line thickness
        displayNode.SetVisibility(True)        # Make sure the line is visible
        displayNode.SetHandlesInteractive(True)  # Disable manipulators (handles)
        displayNode.SetOpacity(0.5)

        # Optionally hide label or set text
        displayNode.SetTextScale(0.5)  # Set small scale; set to 0 if you want it completely invisible

def centerTransform(transform, offsetX=0.0, offsetZ=0.0, rotateClockwise=False):
    """
    Center the transform in the middle of the loaded volume and apply optional offsets and rotation.
    :param transform: Transform node to modify.
    :param offsetX: Optional offset along the X-axis.
    :param offsetZ: Optional offset along the Z-axis.
    :param rotateClockwise: Whether to rotate clockwise or counterclockwise (True = clockwise).
    """
    try:
        volumeNode = slicer.util.getNode("vtkMRMLScalarVolumeNode*")
    except slicer.util.MRMLNodeNotFoundException:
        slicer.util.errorDisplay("Volume node not found. Please load a volume.")
        return

    # Calculate the center of the volume
    bounds = [0.0] * 6
    volumeNode.GetRASBounds(bounds)
    volumeCenter = [
        (bounds[0] + bounds[1]) / 2.0,  # X-center
        (bounds[2] + bounds[3]) / 2.0,  # Y-center
        (bounds[4] + bounds[5]) / 2.0,  # Z-center
    ]
    print(f"[Debug] Volume center: {volumeCenter}")

    # Create a vtkTransform object
    vtk_transform = vtk.vtkTransform()
    vtk_transform.Translate(
        volumeCenter[0] + offsetX, 
        volumeCenter[1], 
        volumeCenter[2] + offsetZ
    )  # Apply translation relative to the volume center

    # Apply rotation in the XZ plane
    angle = 45.0 if rotateClockwise else -45.0  # Rotate +/- 45 degrees
    vtk_transform.RotateY(angle)  # Rotation around the Y-axis (affecting XZ plane)

    # Set the transform to the transform node
    transformMatrix = vtk.vtkMatrix4x4()
    vtk_transform.GetMatrix(transformMatrix)
    transform.SetMatrixTransformToParent(transformMatrix)
    print(f"[Debug] Transform matrix set for '{transform.GetName()}': {transformMatrix}")


def enableItra(transform, segmentNode, segmentName, state):
    """
    Enable or disable interaction (manipulators) for a transform node based on the checkbox state.
    If the corresponding segment is not added or visible, display a message.

    :param transform: The transform node to enable interaction for.
    :param segmentNode: The segment node associated with the transform.
    :param segmentName: The name of the segment (for debugging purposes).
    :param state: Boolean indicating whether to enable (True) or disable (False) interaction.
    """
    if not segmentNode or not segmentNode.GetDisplayNode():
        slicer.util.errorDisplay(f"Segment '{segmentName}' is not added. Please add the segment first.")
        return

    segmentDisplayNode = segmentNode.GetDisplayNode()
    if not segmentDisplayNode.GetVisibility():
        slicer.util.errorDisplay(f"Segment '{segmentName}' is not visible. Please enable it first.")
        return

    if not transform:
        slicer.util.errorDisplay("Transform node is not available.")
        return

    transform.CreateDefaultDisplayNodes()
    displayNode = transform.GetDisplayNode()
    if not displayNode:
        slicer.util.errorDisplay("Transform display node is not available.")
        return

    # Set the visibility of manipulators based on the checkbox state
    displayNode.SetEditorVisibility(state)
    status = "enabled" if state else "disabled"
    print(f"[Debug] Manipulators {status} for '{segmentName}'.")

def addTr(diameter, side):
    """
    Add or update a Tr sphere for the specified side ('L' or 'R') with updated positions from the second version.
    """
    global L_Tr_Node, R_Tr_Node, L_Tr_Transform, R_Tr_Transform

    node, transform = None, None
    name, color, offsetX = "", [], 0.0

    # Define parameters based on the side
    if side == "L":
        node, transform = L_Tr_Node, L_Tr_Transform
        name, color = "L Trochanter", [0, 1, 0]  # Green
        offsetX = -80.0  # Second version's left position
    elif side == "R":
        node, transform = R_Tr_Node, R_Tr_Transform
        name, color = "R Trochanter", [0, 0, 1]  # Blue
        offsetX = 80.0  # Second version's right position

    # Create sphere geometry
    spherePolyData = createSphere(diameter)
    if not spherePolyData:
        slicer.util.errorDisplay(f"Failed to generate {name}.")
        return

    # Create or update the segmentation node
    if not node:
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", name)
        segmentationDisplayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationDisplayNode")
        node.SetAndObserveDisplayNodeID(segmentationDisplayNode.GetID())
        segmentationDisplayNode.SetVisibility3D(True)
        segmentationDisplayNode.SetVisibility2DFill(True)
        segmentationDisplayNode.SetVisibility2DOutline(True)

    # Create or update the transform node
    if not transform:
        transform = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode", f"{name} Transform")
    centerTransform(transform, offsetX=offsetX)  # Only offsetX is updated from the second version
    node.SetAndObserveTransformNodeID(transform.GetID())
    print(f"[Debug] Segment '{name}' observes transform: {transform.GetName()}")

    # Update the segment
    node.GetSegmentation().RemoveAllSegments()
    segment = slicer.vtkSegment()
    segment.SetName(name)
    segment.SetColor(color)
    segment.AddRepresentation(
        slicer.vtkSegmentationConverter.GetClosedSurfaceRepresentationName(),
        spherePolyData
    )
    node.GetSegmentation().AddSegment(segment)

    # Update the label
    addOrUpdateSegmentLabel(
        segmentationNode=node,
        segmentID=name,
        labelName=f"{side}TrLabel",
        textOffset=-20,
        textScale=3.0,
        textColor=color,
    )

    # Update global references
    if side == "L":
        L_Tr_Node, L_Tr_Transform = node, transform
    elif side == "R":
        R_Tr_Node, R_Tr_Transform = node, transform


def addOrUpdateCup(diameter, side):
    """
    Add or update a cup segment (left or right) with the specified diameter.
    This includes creating or updating the segmentation and transform nodes
    while preserving the current position and rotation if the segment already exists.

    :param diameter: Diameter of the cup in millimeters.
    :param side: "L" for left or "R" for right.
    """
    global L_Cup_Node, R_Cup_Node, L_Cup_Transform, R_Cup_Transform

    # Determine which cup to modify
    node, transform = None, None
    name, color, offsetX, offsetZ, rotateY = "", [], 0.0, 20.0, 0.0

    if side == "L":
        node, transform = L_Cup_Node, L_Cup_Transform
        name, color = "L Cup", [0, 1, 0]  # Zielony
        offsetX = -50.0  # Domyślne przesunięcie w osi X
        rotateY = 135.0  # Domyślny obrót w osi Y
    elif side == "R":
        node, transform = R_Cup_Node, R_Cup_Transform
        name, color = "R Cup", [0, 0, 1]  # Niebieski
        offsetX = 50.0  # Domyślne przesunięcie w osi X
        rotateY = -135.0  # Domyślny obrót w osi Y

    # Create cup geometry
    cupPolyData = createCup(diameter, left=(side == "L"))
    if not cupPolyData:
        slicer.util.errorDisplay(f"Failed to generate {name}.")
        return

    # Ensure the transform node exists and preserve its current matrix if possible
    if transform:
        currentTransformMatrix = vtk.vtkMatrix4x4()
        transform.GetMatrixTransformToParent(currentTransformMatrix)
    else:
        transform = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode", f"{name} Transform")
        positionTransform(transform, offsetX=offsetX, offsetZ=offsetZ, rotateY=rotateY)
        currentTransformMatrix = None  # No existing matrix to preserve

    # Ensure the segmentation node exists
    if not node:
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", f"{name}")
        segmentationDisplayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationDisplayNode")
        node.SetAndObserveDisplayNodeID(segmentationDisplayNode.GetID())
        segmentationDisplayNode.SetVisibility3D(True)
        segmentationDisplayNode.SetVisibility2DFill(True)
        segmentationDisplayNode.SetVisibility2DOutline(True)
        node.SetAndObserveTransformNodeID(transform.GetID())
        print(f"[Debug] Created new segmentation '{name}' with transform '{transform.GetName()}'.")
    else:
        print(f"[Debug] Updating existing segmentation '{name}'.")

    # Update the segmentation node with the new cup geometry
    node.GetSegmentation().RemoveAllSegments()
    segment = slicer.vtkSegment()
    segment.SetName(name)
    segment.SetColor(color)
    segment.AddRepresentation(
        slicer.vtkSegmentationConverter.GetClosedSurfaceRepresentationName(),
        cupPolyData
    )
    node.GetSegmentation().AddSegment(segment)

    # Restore the original transform matrix if it exists
    if currentTransformMatrix:
        transform.SetMatrixTransformToParent(currentTransformMatrix)

    # Add or update the label for the segment
    addOrUpdateSegmentLabel(
        segmentationNode=node,
        segmentID=name,
        labelName=f"{side}CupLabel",
        textOffset=-10,
        textScale=3.0,
        textColor=color,
    )

    # Update global variables
    if side == "L":
        L_Cup_Node, L_Cup_Transform = node, transform
    elif side == "R":
        R_Cup_Node, R_Cup_Transform = node, transform

    print(f"[Debug] {name} segment added/updated with diameter {diameter} mm.")


def addOrUpdateSegmentLabel(segmentationNode, segmentID, labelName, textOffset=-5, textScale=3.0, textColor=None):
    """
    Create or update a label for a segment, ensuring the label is positioned correctly 
    and follows the transform of the segment. For "Cup" segments, adjust the label 5 cm above the center.
    """
    if not segmentationNode or not segmentID:
        print(f"[Label Error] Invalid segmentation node or segment ID for '{labelName}'.")
        return

    # Get the global center of the segment
    globalCenter = getSegmentCenter(segmentationNode, segmentID)
    if globalCenter == [0.0, 0.0, 0.0]:
        print(f"[Label Error] Cannot calculate center for segment '{segmentID}'.")
        return
    print(f"[Debug] Global center for segment '{segmentID}': {globalCenter}")

    # Determine specific offset for "Cup" segments
    if "Cup" in segmentID:
        textOffset += 40  # Adjust Z position by 5 cm for Cup segments

    # Adjust position for the label (Z-offset)
    globalLabelPosition = [
        globalCenter[0],
        globalCenter[1],
        globalCenter[2] + textOffset
    ]

    # Transform to local coordinates if the segmentation has a transform
    transformNode = segmentationNode.GetParentTransformNode()
    if transformNode:
        transformMatrix = vtk.vtkMatrix4x4()
        transformNode.GetMatrixTransformToWorld(transformMatrix)
        # Invert the transform to convert global to local
        invertedMatrix = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(transformMatrix, invertedMatrix)

        # Apply the inverted transform to calculate local label position
        localLabelPosition = [0.0, 0.0, 0.0, 1.0]
        globalLabelPosition.append(1.0)  # Add homogeneous coordinate
        invertedMatrix.MultiplyPoint(globalLabelPosition, localLabelPosition)
        localLabelPosition = localLabelPosition[:3]  # Remove the homogeneous component
    else:
        # No transform, local position is the same as global
        localLabelPosition = globalLabelPosition

    print(f"[Debug] Local label position for '{labelName}': {localLabelPosition}")

    # Create or update the label
    fiducialNode = slicer.mrmlScene.GetFirstNodeByName(labelName)
    if fiducialNode:
        fiducialNode.SetNthControlPointPosition(0, *localLabelPosition)
        print(f"[Debug] Updated label '{labelName}' position: {localLabelPosition}")
    else:
        fiducialNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", labelName)
        fiducialNode.AddControlPoint(localLabelPosition, segmentationNode.GetName())
        print(f"[Debug] Created label '{labelName}' at position: {localLabelPosition}")

    # Customize display properties
    displayNode = fiducialNode.GetDisplayNode()
    if displayNode:
        displayNode.SetGlyphScale(0)  # Hide glyph marker
        displayNode.SetTextScale(textScale)
        if textColor:
            displayNode.SetColor(*textColor)
        displayNode.SetVisibility(True)
        displayNode.SetVisibility3D(True)
        print(f"[Debug] Display properties set for label '{labelName}': Scale={textScale}, Color={textColor}")
    else:
        print(f"[Label Error] No display node for '{labelName}'.")

    # Assign transform to the label
    if transformNode:
        fiducialNode.SetAndObserveTransformNodeID(transformNode.GetID())
        print(f"[Debug] Label '{labelName}' observes transform: {transformNode.GetName()}")


def scaleCurrentSegmentationBoundingBox(increment=True, label=None):
    """
    Adjust the size of the imported segmentation bounding box by adding a calculated step to the Z-axis size,
    while scaling other dimensions proportionally. Automatically sets the smallest predefined size
    for imported STL if no valid size is currently set. Updates the label dynamically.

    Parameters:
        increment (bool): If True, scale up; if False, scale down.
        label (QLabel): QLabel widget to display the current size name (e.g., "6", "7.5").
    """

    # Mapa rozmiarów: powiązanie nazw rozmiarów z wartościami Z
    sizeMapping = {
        6: 97.5,
        7.5: 99,
        9: 102.5,
        10: 105,
        11: 107.5,
        12.5: 109,
        13.5: 111,
        15: 115,
        17.5: 119,
        20: 125
    }

    # Lista nazw rozmiarów i ich wartości
    sizeNames = list(sizeMapping.keys())
    sizeValues = list(sizeMapping.values())

    global currentSegmentationNode, currentSegmentID, currentPinSize, maxPinSize, originalPinSize

    if not currentSegmentationNode or not currentSegmentID:
        slicer.util.errorDisplay("No current segmentation to scale.")
        return

    if originalPinSize is None or maxPinSize is None:
        slicer.util.errorDisplay("Scaling limits not properly initialized.")
        return

    # Automatyczne przypisanie najmniejszej wartości z listy jako początkowego rozmiaru
    if currentPinSize is None or currentPinSize not in sizeValues:
        currentPinSize = sizeValues[0]  # Ustaw najmniejszy rozmiar z listy
        print(f"Initial size set to {currentPinSize} mm.")

    # Znalezienie aktualnego indeksu w liście rozmiarów
    try:
        currentIndex = sizeValues.index(currentPinSize)
    except ValueError:
        slicer.util.errorDisplay("Current size does not match any predefined size.")
        return

    # Obliczenie nowego indeksu w zależności od kierunku (zwiększenie/zmniejszenie)
    newIndex = currentIndex + (1 if increment else -1)

    # Sprawdzenie limitów indeksów
    if newIndex < 0:
        slicer.util.errorDisplay("Minimum segmentation size reached.")
        return
    if newIndex >= len(sizeValues):
        slicer.util.errorDisplay("Maximum segmentation size reached.")
        return

    # Pobranie nowego rozmiaru i obliczenie współczynnika skalowania
    newPinSizeZ = sizeValues[newIndex]
    scaleFactorZ = newPinSizeZ / currentPinSize

    # Pobierz węzeł transformacji przypisany do segmentacji
    transformNodeID = currentSegmentationNode.GetTransformNodeID()
    if not transformNodeID:
        slicer.util.errorDisplay("No transform node assigned to the current segmentation.")
        return

    transformNode = slicer.mrmlScene.GetNodeByID(transformNodeID)
    if not transformNode:
        slicer.util.errorDisplay("Transform node not found.")
        return

    # Zachowaj istniejącą translację i rotację
    originalMatrix = vtk.vtkMatrix4x4()
    transformNode.GetMatrixTransformToParent(originalMatrix)

    # Wyodrębnij istniejącą translację
    translation = [
        originalMatrix.GetElement(0, 3),  # X translation
        originalMatrix.GetElement(1, 3),  # Y translation
        originalMatrix.GetElement(2, 3),  # Z translation
    ]

    # Zastosuj skalowanie
    vtk_transform = vtk.vtkTransform()
    vtk_transform.Translate(translation)  # Najpierw translacja
    vtk_transform.Scale(scaleFactorZ, scaleFactorZ, scaleFactorZ)  # Skalowanie proporcjonalne
    vtk_transform.Translate(-translation[0], -translation[1], -translation[2])  # Cofnięcie translacji
    vtk_transform.Concatenate(originalMatrix)  # Zastosowanie oryginalnej macierzy

    # Aktualizacja węzła transformacji z nową macierzą
    scaledMatrix = vtk.vtkMatrix4x4()
    vtk_transform.GetMatrix(scaledMatrix)
    transformNode.SetMatrixTransformToParent(scaledMatrix)

    # Aktualizacja bieżącego rozmiaru w osi Z
    currentPinSize = newPinSizeZ

    # Aktualizacja etykiety
    newSizeName = sizeNames[newIndex]  # Pobierz nazwę rozmiaru
    addOrUpdateDynamicSegmentLabel(
        segmentationNode=currentSegmentationNode,
        segmentID=currentSegmentID,
        labelText=str(newSizeName),  # Tekst etykiety
        textOffset=-5,
        textScale=3.0,
        textColor=(1.0, 1.0, 1.0)  # Biały kolor
    )
    print(f"Label updated to: Size {newSizeName}")

    # Odświeżenie widoków
    slicer.app.processEvents()
    print(f"Segmentation size updated to {currentPinSize} mm in Z-axis.")

    
def scaleCurrentSegmentationBoundingBox(increment=True, label=None):
    """
    Adjust the size of the imported segmentation bounding box by scaling the Z-axis
    and scaling other dimensions proportionally. Automatically sets the smallest predefined size
    for imported STL if no valid size is currently set. Updates the label dynamically.

    Parameters:
        increment (bool): If True, scale up; if False, scale down.
        label (QLabel): QLabel widget to display the current size name (e.g., "6", "7.5").
    """
    global currentSegmentationNode, currentSegmentID, currentPinSize, maxPinSize, originalPinSize, sizeMapping

    if not currentSegmentationNode or not currentSegmentID:
        slicer.util.errorDisplay("No current segmentation to scale.")
        return

    if originalPinSize is None or maxPinSize is None:
        slicer.util.errorDisplay("Scaling limits not properly initialized.")
        return

    if sizeMapping:
        sizeNames = list(sizeMapping.keys())
        sizeValues = [entry['value'] for entry in sizeMapping.values()]
        
        # Automatyczne przypisanie najmniejszej wartości z listy jako początkowego rozmiaru
        if currentPinSize is None or currentPinSize not in sizeValues:
            currentPinSize = sizeValues[0]
            print(f"Initial size set to {currentPinSize} mm.")

        try:
            currentIndex = sizeValues.index(currentPinSize)
        except ValueError:
            slicer.util.errorDisplay("Current size does not match any predefined size.")
            return

        newIndex = currentIndex + (1 if increment else -1)

        if newIndex < 0:
            slicer.util.errorDisplay("Minimum segmentation size reached.")
            return
        if newIndex >= len(sizeValues):
            slicer.util.errorDisplay("Maximum segmentation size reached.")
            return

        newPinSizeZ = sizeValues[newIndex]
        scaleFactor = newPinSizeZ / currentPinSize
        newSizeName = sizeNames[newIndex]
    else:
        step = 1 if increment else -1
        newPinSizeZ = currentPinSize + step
        if newPinSizeZ <= 0:
            slicer.util.errorDisplay("Minimum size reached.")
            return

        scaleFactor = newPinSizeZ / currentPinSize
        newSizeName = f"{newPinSizeZ:.1f}"

    transformNodeID = currentSegmentationNode.GetTransformNodeID()
    if not transformNodeID:
        slicer.util.errorDisplay("No transform node assigned to the current segmentation.")
        return

    transformNode = slicer.mrmlScene.GetNodeByID(transformNodeID)
    if not transformNode:
        slicer.util.errorDisplay("Transform node not found.")
        return

    originalMatrix = vtk.vtkMatrix4x4()
    transformNode.GetMatrixTransformToParent(originalMatrix)

    translation = [
        originalMatrix.GetElement(0, 3),
        originalMatrix.GetElement(1, 3),
        originalMatrix.GetElement(2, 3),
    ]

    vtk_transform = vtk.vtkTransform()
    vtk_transform.Translate(translation)
    vtk_transform.Scale(scaleFactor, scaleFactor, scaleFactor)  # Skalowanie proporcjonalne
    vtk_transform.Translate(-translation[0], -translation[1], -translation[2])
    vtk_transform.Concatenate(originalMatrix)

    scaledMatrix = vtk.vtkMatrix4x4()
    vtk_transform.GetMatrix(scaledMatrix)
    transformNode.SetMatrixTransformToParent(scaledMatrix)

    currentPinSize = newPinSizeZ

    addOrUpdateDynamicSegmentLabel(
        segmentationNode=currentSegmentationNode,
        segmentID=currentSegmentID,
        labelText=str(newSizeName),
        textOffset=-5,
        textScale=3.0,
        textColor=(1.0, 1.0, 1.0)
    )
    print(f"Label updated to: Size {newSizeName}")
    slicer.app.processEvents()
    print(f"Segmentation size updated to {currentPinSize} mm.")



def positionTransform(transform, offsetX=0.0, offsetZ=0.0, rotateY=0.0):
    """
    Pozycjonuje transform w środku wolumenu z opcjonalnym przesunięciem i rotacją w osi Y.
    :param transform: Węzeł transformacji do modyfikacji.
    :param offsetX: Przesunięcie w osi X.
    :param offsetZ: Przesunięcie w osi Z.
    :param rotateY: Kąt obrotu w osi Y (w stopniach).
    """
    try:
        volumeNode = slicer.util.getNode("vtkMRMLScalarVolumeNode*")
    except slicer.util.MRMLNodeNotFoundException:
        slicer.util.errorDisplay("Nie znaleziono węzła wolumenu. Proszę wczytać wolumen.")
        return

    # Obliczanie środka wolumenu
    bounds = [0.0] * 6
    volumeNode.GetRASBounds(bounds)
    volumeCenter = [
        (bounds[0] + bounds[1]) / 2.0,  # X-środek
        (bounds[2] + bounds[3]) / 2.0,  # Y-środek
        (bounds[4] + bounds[5]) / 2.0,  # Z-środek
    ]
    print(f"[Debug] Środek wolumenu: {volumeCenter}")

    # Tworzenie vtkTransform do modyfikacji
    vtk_transform = vtk.vtkTransform()
    vtk_transform.Translate(
        volumeCenter[0] + offsetX,
        volumeCenter[1],  # Brak przesunięcia w osi Y
        volumeCenter[2] + offsetZ
    )
    vtk_transform.RotateY(rotateY)  # Obrót w osi Y

    # Ustawienie macierzy transformacji
    transformMatrix = vtk.vtkMatrix4x4()
    vtk_transform.GetMatrix(transformMatrix)
    transform.SetMatrixTransformToParent(transformMatrix)

    print(f"[Debug] Transformacja z ustawioną macierzą: {transform.GetName()} - Obrót Y: {rotateY}, Przesunięcie X: {offsetX}, Z: {offsetZ}")
    
def toggleCupVisibilityOrAdd(diameter, side):
    """
    Toggle visibility of the specified Cup (L/R) along with its label. Add Cup if it doesn't exist.
    If the segment is being hidden, its manipulators and spinbox will also be turned off.
    """
    global L_Cup_Node, R_Cup_Node, L_Cup_Transform, R_Cup_Transform, cupSizeSpinBoxL, cupSizeSpinBoxR

    node, transform, name, labelName, spinbox = None, None, "", "", None
    if side == "L":
        node, transform, spinbox = L_Cup_Node, L_Cup_Transform, cupSizeSpinBoxL
        name, labelName = "Left Cup", "LCupLabel"
    elif side == "R":
        node, transform, spinbox = R_Cup_Node, R_Cup_Transform, cupSizeSpinBoxR
        name, labelName = "Right Cup", "RCupLabel"

    # Sprawdź, czy załadowano wolumen
    volumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
    if not volumeNode:
        slicer.util.errorDisplay(
            "No volume loaded. Please import an X-ray or CT volume using the 'Import' buttons."
        )
        return  # Zakończ funkcję, jeśli brak wolumenu

    if not node:
        # Jeśli segment nie istnieje, dodaj go
        addOrUpdateCup(diameter, side)
        # Upewnij się, że spinbox jest aktywny
        if spinbox:
            spinbox.setEnabled(True)
    else:
        displayNode = node.GetDisplayNode()
        if not displayNode:
            slicer.util.errorDisplay(f"{name} display node not found.")
            return

        # Sprawdź aktualny stan widoczności i przełącz
        currentVisibility = displayNode.GetVisibility()
        newVisibility = not currentVisibility
        displayNode.SetVisibility(newVisibility)

        # Ukryj manipulatory, jeśli segment jest ukrywany
        if not newVisibility and transform:
            transform.CreateDefaultDisplayNodes()
            transformDisplayNode = transform.GetDisplayNode()
            if transformDisplayNode:
                transformDisplayNode.SetEditorVisibility(False)  # Wyłącz manipulatory

        # Ukryj label, jeśli segment jest ukrywany
        fiducialNode = slicer.mrmlScene.GetFirstNodeByName(labelName)
        if fiducialNode:
            fiducialDisplayNode = fiducialNode.GetDisplayNode()
            if fiducialDisplayNode:
                fiducialDisplayNode.SetVisibility(newVisibility)

        # Dezaktywuj spinbox, jeśli segment jest ukrywany
        if spinbox:
            spinbox.setEnabled(newVisibility)
            if not newVisibility:
                spinbox.blockSignals(True)  # Wyłącz sygnały podczas resetu
                spinbox.setValue(50)  # Przywróć domyślną wartość
                spinbox.blockSignals(False)

def toggleTrVisibilityOrAdd(diameter, side):
    """
    Toggle visibility of the specified Tr (L/R) along with its label. Add Tr if it doesn't exist.
    If the segment is being hidden, its manipulators will also be turned off.
    """
    global L_Tr_Node, R_Tr_Node, L_Tr_Transform, R_Tr_Transform

    node, transform, name, labelName = None, None, "", ""
    if side == "L":
        node, transform = L_Tr_Node, L_Tr_Transform
        name, labelName = "Left Tr", "LTrLabel"
    elif side == "R":
        node, transform = R_Tr_Node, R_Tr_Transform
        name, labelName = "Right Tr", "RTrLabel"

    # Sprawdź, czy załadowano wolumen
    volumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
    if not volumeNode:
        slicer.util.errorDisplay(
            "No volume loaded. Please import an X-ray or CT volume using the 'Import' buttons."
        )
        return  # Zakończ funkcję, jeśli brak wolumenu

    if not node:
        # Jeśli segment nie istnieje, dodaj go
        addTr(diameter, side)
    else:
        displayNode = node.GetDisplayNode()
        if not displayNode:
            slicer.util.errorDisplay(f"{name} display node not found.")
            return

        # Sprawdź aktualny stan widoczności i przełącz
        currentVisibility = displayNode.GetVisibility()
        newVisibility = not currentVisibility
        displayNode.SetVisibility(newVisibility)

        # Ukryj manipulatory, jeśli segment jest ukrywany
        if not newVisibility and transform:
            transform.CreateDefaultDisplayNodes()
            transformDisplayNode = transform.GetDisplayNode()
            if transformDisplayNode:
                transformDisplayNode.SetEditorVisibility(False)  # Wyłącz manipulatory

        # Ukryj label, jeśli segment jest ukrywany
        fiducialNode = slicer.mrmlScene.GetFirstNodeByName(labelName)
        if fiducialNode:
            fiducialDisplayNode = fiducialNode.GetDisplayNode()
            if fiducialDisplayNode:
                fiducialDisplayNode.SetVisibility(newVisibility)

def addOrUpdateHeadL():
    global Head_Transform_L, L_Cup_Transform, L_Tr_Transform, R_Tr_Transform, Ischial_Line_Node

    # Check if there's already a right head, and hide it
    rightHeadNode = slicer.mrmlScene.GetFirstNodeByName("Head R Segmentation")
    if rightHeadNode:
        rightHeadDisplayNode = rightHeadNode.GetDisplayNode()
        if rightHeadDisplayNode:
            rightHeadDisplayNode.SetVisibility(False)

    """
    Add or update a head sphere for the left side (L) with Ischial line as the reference.
    """
    global Head_Transform_L, L_Cup_Transform, L_Tr_Transform, R_Tr_Transform, Ischial_Line_Node

    if not isVolumeLoaded():
        slicer.util.errorDisplay("No volume loaded. Please load a volume to add segmentations.")
        return

    # Check if Ischial line exists
    if not Ischial_Line_Node:
        slicer.util.errorDisplay("Ischial line is missing. Please add it before calculating Head L.")
        return

    # Get transforms for L Cup, L Tr, and R Tr
    cupTransform = L_Cup_Transform
    lTrTransform = L_Tr_Transform
    rTrTransform = R_Tr_Transform

    if not cupTransform or not lTrTransform or not rTrTransform:
        slicer.util.errorDisplay("Cannot place Head L because required transforms are missing.")
        return

    # Get Z-coordinate of the Ischial line control points
    lineStart = [0.0, 0.0, 0.0]
    lineEnd = [0.0, 0.0, 0.0]
    Ischial_Line_Node.GetNthControlPointPositionWorld(0, lineStart)
    Ischial_Line_Node.GetNthControlPointPositionWorld(1, lineEnd)

    # Find intersection points for L Tr and R Tr
    def findIntersectionZ(transformNode, lineStart, lineEnd):
        """
        Find the Z-coordinate of the intersection of the Z-axis of the transform with the Ischial line.
        """
        # Pobierz współrzędne X i Y transformacji
        matrix = vtk.vtkMatrix4x4()
        transformNode.GetMatrixTransformToParent(matrix)
        xTransform = matrix.GetElement(0, 3)
        yTransform = matrix.GetElement(1, 3)

        # Współrzędne punktów linii kulszowej
        x1, y1, z1 = lineStart
        x2, y2, z2 = lineEnd

        # Oblicz współczynnik parametryczny t
        t = ((xTransform - x1) * (x2 - x1) + (yTransform - y1) * (y2 - y1)) / (
            (x2 - x1) ** 2 + (y2 - y1) ** 2
        )

        # Ogranicz t do zakresu [0, 1]
        t = max(0, min(1, t))

        # Oblicz współrzędną Z punktu przecięcia
        intersectionZ = z1 + t * (z2 - z1)
        return intersectionZ

    # Oblicz punkty przecięcia dla L Tr i R Tr
    lTrIntersectionZ = findIntersectionZ(lTrTransform, lineStart, lineEnd)
    rTrIntersectionZ = findIntersectionZ(rTrTransform, lineStart, lineEnd)

    # Pobierz Z transformacji L Tr i R Tr
    def getTransformCenterZ(transformNode):
        matrix = vtk.vtkMatrix4x4()
        transformNode.GetMatrixTransformToParent(matrix)
        return matrix.GetElement(2, 3)

    lTrZ = getTransformCenterZ(lTrTransform)
    rTrZ = getTransformCenterZ(rTrTransform)

    # Oblicz odległości segmentów w osi Z
    segmentLTrLength = abs(lTrZ - lTrIntersectionZ)  # Odległość od L Tr do punktu przecięcia
    segmentRTrLength = abs(rTrZ - rTrIntersectionZ)  # Odległość od R Tr do punktu przecięcia

    # Oblicz Z-pozycję dla Head L względem L Cup
    cupZ = getTransformCenterZ(cupTransform)
    headZ = cupZ + (segmentLTrLength - segmentRTrLength)

    # Pobierz X, Y współrzędne z L Cup transform
    matrix = vtk.vtkMatrix4x4()
    cupTransform.GetMatrixTransformToParent(matrix)
    headX = matrix.GetElement(0, 3)  # X współrzędna
    headY = matrix.GetElement(1, 3)  # Y współrzędna

    # Stwórz kulę o średnicy 28 mm
    spherePolyData = createSphere(28)
    if not spherePolyData:
        slicer.util.errorDisplay("Failed to generate Head L.")
        return

    # Stwórz lub zaktualizuj węzeł segmentacji dla Head L
    segmentationNode = slicer.mrmlScene.GetFirstNodeByName("Head L Segmentation")
    if not segmentationNode:
        segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "Head L Segmentation")
        segmentationDisplayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationDisplayNode")
        segmentationNode.SetAndObserveDisplayNodeID(segmentationDisplayNode.GetID())

        segmentationDisplayNode.SetVisibility3D(True)
        segmentationDisplayNode.SetVisibility2DFill(True)
        segmentationDisplayNode.SetVisibility2DOutline(True)

    # Upewnij się, że transformacja istnieje
    transformNodeID = segmentationNode.GetTransformNodeID()
    if transformNodeID:
        transformNode = slicer.mrmlScene.GetNodeByID(transformNodeID)
    else:
        # Stwórz nowy węzeł transformacji, jeśli nie istnieje
        transformNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode", "Head L Transform")
        segmentationNode.SetAndObserveTransformNodeID(transformNode.GetID())

    # Zaktualizuj globalne odniesienie do węzła transformacji
    Head_Transform_L = transformNode

    # Ustaw transformację dla Head L
    vtk_transform = vtk.vtkTransform()
    vtk_transform.Translate(headX, headY, headZ)  # Ustaw pozycję
    transformMatrix = vtk.vtkMatrix4x4()
    vtk_transform.GetMatrix(transformMatrix)
    transformNode.SetMatrixTransformToParent(transformMatrix)

    # Zaktualizuj segmentację z kulą
    segmentationNode.GetSegmentation().RemoveAllSegments()
    segment = slicer.vtkSegment()
    segment.SetName("Head L")
    segment.SetColor([1, 0, 0])  # Czerwony kolor
    segment.AddRepresentation(
        slicer.vtkSegmentationConverter.GetClosedSurfaceRepresentationName(),
        spherePolyData
    )
    segmentationNode.GetSegmentation().AddSegment(segment)

    # Po opcjonalnym wywołaniu displayDynamicLength()
    # Wyświetlanie odległości dla Head L, z uwzględnieniem korekcji (jeśli dostępna)
    correctedShift_L = left_vertical_shift if rotationalCorrectionEnabled else 0
    displayDistanceZ("L", Head_Transform_L, L_Tr_Transform, correctedShift=correctedShift_L)

def addOrUpdateHeadR():
    
    global Head_Transform_R, R_Cup_Transform, R_Tr_Transform, L_Tr_Transform, Ischial_Line_Node

    # Check if there's already a left head, and hide it
    leftHeadNode = slicer.mrmlScene.GetFirstNodeByName("Head L Segmentation")
    if leftHeadNode:
        leftHeadDisplayNode = leftHeadNode.GetDisplayNode()
        if leftHeadDisplayNode:
            leftHeadDisplayNode.SetVisibility(False)

    # Continue with the rest of the `addOrUpdateHeadR` function...
    # Existing implementation goes here

    
    """
    Add or update a head sphere for the right side (R) with Ischial line as the reference.
    """
    global Head_Transform_R, R_Cup_Transform, R_Tr_Transform, L_Tr_Transform, Ischial_Line_Node

    if not isVolumeLoaded():
        slicer.util.errorDisplay("No volume loaded. Please load a volume to add segmentations.")
        return

    # Check if Ischial line exists
    if not Ischial_Line_Node:
        slicer.util.errorDisplay("Ischial line is missing. Please add it before calculating Head R.")
        return

    # Get transforms for R Cup and R Tr
    cupTransform = R_Cup_Transform
    rTrTransform = R_Tr_Transform

    if not cupTransform or not rTrTransform:
        slicer.util.errorDisplay("Cannot place Head R because required transforms are missing.")
        return

    # Optional: Handle missing left transform only if needed for calculations
    if not L_Tr_Transform:
        print("[Warning] L_Tr_Transform is None. Skipping dependent calculations.")

    # Get Z-coordinates from transform nodes
    def getTransformCenterZ(transformNode):
        if not isinstance(transformNode, slicer.vtkMRMLTransformNode):
            print(f"[Warning] Node {transformNode.GetName()} is not a transform node.")
            return None
        matrix = vtk.vtkMatrix4x4()
        transformNode.GetMatrixTransformToParent(matrix)
        return matrix.GetElement(2, 3)

    # Get Z-coordinate of the Ischial line control points
    def getIschialLineControlPointZ(pointIndex):
        try:
            point = [0.0, 0.0, 0.0]
            Ischial_Line_Node.GetNthControlPointPositionWorld(pointIndex, point)
            return point[2]  # Z-coordinate of the control point
        except Exception as e:
            print(f"[Warning] Error accessing Ischial line control points: {e}")
            return None

    # Get Z positions
    cupZ = getTransformCenterZ(cupTransform)
    rTrZ = getTransformCenterZ(rTrTransform)
    lineStartZ = getIschialLineControlPointZ(0)
    lineEndZ = getIschialLineControlPointZ(1)

    if None in [cupZ, rTrZ, lineStartZ, lineEndZ]:
        print("[Warning] Missing data for Head R calculation. Skipping...")
        return

    # Calculate the lengths of the segments in Z
    segmentRTrLength = abs(lineEndZ - rTrZ)
    segmentLTrLength = 0  # Set to 0 if L_Tr_Transform is missing
    if L_Tr_Transform:
        lTrZ = getTransformCenterZ(L_Tr_Transform)
        segmentLTrLength = abs(lineStartZ - lTrZ)

    # Calculate Z position for Head R relative to R Cup
    headZ = cupZ + (segmentRTrLength - segmentLTrLength)

    # Get X, Y coordinates from R Cup transform
    matrix = vtk.vtkMatrix4x4()
    cupTransform.GetMatrixTransformToParent(matrix)
    headX = matrix.GetElement(0, 3)
    headY = matrix.GetElement(1, 3)

    # Create a 28 mm diameter sphere
    spherePolyData = createSphere(28)
    if not spherePolyData:
        slicer.util.errorDisplay("Failed to generate Head R.")
        return

    # Create or update the segmentation node for Head R
    segmentationNode = slicer.mrmlScene.GetFirstNodeByName("Head R Segmentation")
    if not segmentationNode:
        segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "Head R Segmentation")
        segmentationDisplayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationDisplayNode")
        segmentationNode.SetAndObserveDisplayNodeID(segmentationDisplayNode.GetID())

        segmentationDisplayNode.SetVisibility3D(True)
        segmentationDisplayNode.SetVisibility2DFill(True)
        segmentationDisplayNode.SetVisibility2DOutline(True)

    # Ensure the transform exists
    transformNodeID = segmentationNode.GetTransformNodeID()
    if transformNodeID:
        transformNode = slicer.mrmlScene.GetNodeByID(transformNodeID)
    else:
        transformNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode", "Head R Transform")
        segmentationNode.SetAndObserveTransformNodeID(transformNode.GetID())

    # Update global reference to the transform node
    Head_Transform_R = transformNode

    # Position the transform for Head R
    vtk_transform = vtk.vtkTransform()
    vtk_transform.Translate(headX, headY, headZ)
    transformMatrix = vtk.vtkMatrix4x4()
    vtk_transform.GetMatrix(transformMatrix)
    transformNode.SetMatrixTransformToParent(transformMatrix)

    # Update segmentation with the sphere
    segmentationNode.GetSegmentation().RemoveAllSegments()
    segment = slicer.vtkSegment()
    segment.SetName("Head R")
    segment.SetColor([1, 0, 0])
    segment.AddRepresentation(
        slicer.vtkSegmentationConverter.GetClosedSurfaceRepresentationName(),
        spherePolyData
    )
    segmentationNode.GetSegmentation().AddSegment(segment)

    print(f"[Debug] Head_Transform_R assigned: {Head_Transform_R.GetName() if Head_Transform_R else 'None'}")

        # Wyświetlanie odległości dla Head R, z uwzględnieniem korekcji (jeśli dostępna)
    correctedShift_R = right_vertical_shift if rotationalCorrectionEnabled else 0
    displayDistanceZ("R", Head_Transform_R, R_Tr_Transform, correctedShift=correctedShift_R)

def toggleHeadVisibilityOrAdd(side):
    """
    Toggle visibility or add a head segment for the specified side ('L' or 'R').
    Includes rotational correction if enabled and ensures proper handling of opposite side.
    Updates reference line only if it exists and is visible.
    """
    global Head_Transform_L, Head_Transform_R, rotationalCorrectionEnabled
    global L_Tr_Node, R_Tr_Node, L_Cup_Transform, R_Cup_Transform
    global left_vertical_shift, right_vertical_shift

    # Determine the appropriate segment and update function
    if side == "L":
        segmentName = "Head L Segmentation"
        oppositeSegmentName = "Head R Segmentation"
        updateHeadFunc = addOrUpdateHeadL
        transform = Head_Transform_L
        lineName = "Upper Reference (Left)"
    elif side == "R":
        segmentName = "Head R Segmentation"
        oppositeSegmentName = "Head L Segmentation"
        updateHeadFunc = addOrUpdateHeadR
        transform = Head_Transform_R
        lineName = "Upper Reference (Right)"
    else:
        slicer.util.errorDisplay("Invalid side specified. Use 'L' or 'R'.")
        return

    # Hide the opposite side
    oppositeHeadNode = slicer.mrmlScene.GetFirstNodeByName(oppositeSegmentName)
    if oppositeHeadNode:
        oppositeDisplayNode = oppositeHeadNode.GetDisplayNode()
        if oppositeDisplayNode:
            oppositeDisplayNode.SetVisibility(False)

    # Check if the current side's head exists
    headNode = slicer.mrmlScene.GetFirstNodeByName(segmentName)
    if not headNode:
        print(f"[Debug] {segmentName} not found. Creating...")
        updateHeadFunc()  # Create the head segment
        return

    # Apply rotational correction if enabled
    z_correction = 0  # Default Z correction
    if rotationalCorrectionEnabled:
        print(f"[Debug] Rotational Correction Enabled: {rotationalCorrectionEnabled}")

        # Get positions of the Left and Right Trochanters
        left_trochanter_pos = get_world_position(L_Tr_Node)
        right_trochanter_pos = get_world_position(R_Tr_Node)
        if left_trochanter_pos is None or right_trochanter_pos is None:
            slicer.util.errorDisplay("Trochanter positions not found.")
            return

        # Get the centers of the Left and Right Cups
        leftcup_center = get_cup_center(L_Cup_Transform)
        rightcup_center = get_cup_center(R_Cup_Transform)
        if leftcup_center is None or rightcup_center is None:
            slicer.util.errorDisplay("Cup centers not found.")
            return

        # Calculate angles
        angle_end, angle_start = calculate_corrected_angles()
        if angle_end is None or angle_start is None:
            slicer.util.errorDisplay("Failed to calculate angles.")
            return

        # Calculate angle difference
        angle_difference = angle_start - angle_end
        print(f"[Debug] Angle Difference: {angle_difference:.2f}°")

        # Calculate corrected positions
        left_corrected_pos, left_vertical_shift = calculate_corrected_position_and_vertical_shift(
            trochanter_pos=np.array(left_trochanter_pos),
            cup_center=np.array(leftcup_center),
            angle_difference=angle_difference
        )
        right_corrected_pos, right_vertical_shift = calculate_corrected_position_and_vertical_shift(
            trochanter_pos=np.array(right_trochanter_pos),
            cup_center=np.array(rightcup_center),
            angle_difference=angle_difference
        )

        # Assign shifts based on the selected side
        z_correction = left_vertical_shift if side == "L" else right_vertical_shift

    # Update head geometry
    print(f"[Debug] Updating {segmentName}.")
    updateHeadFunc()

    # Apply Z correction if needed
    if rotationalCorrectionEnabled and z_correction != 0:
        print(f"[Debug] Applying rotational correction: {z_correction:.2f} mm in Z-axis.")
        matrix = vtk.vtkMatrix4x4()
        transform.GetMatrixTransformToParent(matrix)
        matrix.SetElement(2, 3, matrix.GetElement(2, 3) + z_correction)  # Apply Z correction
        transform.SetMatrixTransformToParent(matrix)

    # Toggle visibility of the current side's head
    displayNode = headNode.GetDisplayNode()
    if displayNode:
        currentVisibility = displayNode.GetVisibility()
        displayNode.SetVisibility(not currentVisibility)
        print(f"[Debug] {segmentName} visibility set to {'ON' if not currentVisibility else 'OFF'}.")

        # Update the reference line only if it exists and is visible
        lineNode = slicer.mrmlScene.GetFirstNodeByName(lineName)
        if lineNode:
            lineDisplayNode = lineNode.GetDisplayNode()
            if lineDisplayNode and lineDisplayNode.GetVisibility():
                print(f"[Debug] Updating reference line for {side}.")
                toggleReferenceLine("Left" if side == "L" else "Right")

def toggleRotationalCorrection(state):
    """
    Toggle the rotational correction mode based on radiobutton state.
    """
    global rotationalCorrectionEnabled
    rotationalCorrectionEnabled = state
    print(f"Rotational Correction Enabled: {rotationalCorrectionEnabled}")

lastSide = None  # Zmienna globalna przechowująca ostatnio wybraną stronę

def updateHead(side):
    """
    Update the geometry and position of the head sphere without toggling its visibility.
    Automatically adds the head segment if it is missing.
    Updates reference line position if it exists and respects its visibility state.
    Includes rotational correction if enabled.
    """
    global Head_Transform_L, Head_Transform_R, rotationalCorrectionEnabled
    global L_Tr_Node, R_Tr_Node, L_Cup_Transform, R_Cup_Transform
    global left_vertical_shift, right_vertical_shift
    global referenceLineButton
    global lastSide


    # Determine the appropriate segment and update function
    if side == "L":
        segmentName = "Head L Segmentation"
        updateHeadFunc = addOrUpdateHeadL
        transform = Head_Transform_L
        lineName = "Upper Reference (Left)"
    elif side == "R":
        segmentName = "Head R Segmentation"
        updateHeadFunc = addOrUpdateHeadR
        transform = Head_Transform_R
        lineName = "Upper Reference (Right)"
    else:
        slicer.util.errorDisplay("Invalid side specified. Use 'L' or 'R'.")
        return

    # Ensure necessary nodes and transforms exist
    if not L_Tr_Node or not R_Tr_Node or not L_Cup_Transform or not R_Cup_Transform:
        slicer.util.errorDisplay("Required nodes or transforms are missing.")
        return

    # Ensure the transform exists, create the head segment if missing
    if not transform:
        print(f"[Debug] {segmentName} transform missing. Adding the head segment.")
        updateHeadFunc()  # Automatically add the head segment
        transform = Head_Transform_L if side == "L" else Head_Transform_R

    # Calculate positions and vertical shifts if correction is enabled
    z_correction = 0  # Default value for no correction
    if rotationalCorrectionEnabled:
        print(f"[Debug] Rotational Correction Enabled: {rotationalCorrectionEnabled}")

        # Get positions of the Left and Right Trochanters
        left_trochanter_pos = get_world_position(L_Tr_Node)
        right_trochanter_pos = get_world_position(R_Tr_Node)
        if left_trochanter_pos is None or right_trochanter_pos is None:
            slicer.util.errorDisplay("Trochanter positions not found.")
            return

        # Get the centers of the Left and Right Cups
        leftcup_center = get_cup_center(L_Cup_Transform)
        rightcup_center = get_cup_center(R_Cup_Transform)
        if leftcup_center is None or rightcup_center is None:
            slicer.util.errorDisplay("Cup centers not found.")
            return

        # Calculate angles
        angle_end, angle_start = calculate_corrected_angles()
        if angle_end is None or angle_start is None:
            slicer.util.errorDisplay("Failed to calculate angles.")
            return

        # Calculate angle difference
        angle_difference = angle_start - angle_end
        print(f"[Debug] Angle Difference: {angle_difference:.2f}°")

        # Calculate corrected positions
        if side == "L":
            _, left_vertical_shift = calculate_corrected_position_and_vertical_shift(
                trochanter_pos=np.array(left_trochanter_pos),
                cup_center=np.array(leftcup_center),
                angle_difference=angle_difference
            )
            z_correction = left_vertical_shift
        else:
            _, right_vertical_shift = calculate_corrected_position_and_vertical_shift(
                trochanter_pos=np.array(right_trochanter_pos),
                cup_center=np.array(rightcup_center),
                angle_difference=angle_difference
            )
            z_correction = right_vertical_shift

        print(f"[Debug] Vertical Shift for {side}: {z_correction:.2f} mm")

    # Call the update function to adjust geometry
    print(f"[Debug] Updating {segmentName}.")
    updateHeadFunc()

    # Apply rotational correction if enabled
    if rotationalCorrectionEnabled and z_correction != 0:
        print(f"[Debug] Applying rotational correction: {z_correction:.2f} mm in Z-axis.")
        matrix = vtk.vtkMatrix4x4()
        transform.GetMatrixTransformToParent(matrix)
        matrix.SetElement(2, 3, matrix.GetElement(2, 3) + z_correction)  # Apply Z correction
        transform.SetMatrixTransformToParent(matrix)

    # Ensure the segmentation visibility remains unchanged
    headNode = slicer.mrmlScene.GetFirstNodeByName(segmentName)
    if headNode:
        displayNode = headNode.GetDisplayNode()
        if displayNode:
            displayNode.SetVisibility(True)
            print(f"[Debug] {segmentName} visibility set to ON.")

    # Update the reference line position if it exists
    lineNode = slicer.mrmlScene.GetFirstNodeByName(lineName)
    if lineNode:
        lineDisplayNode = lineNode.GetDisplayNode()
        if lineDisplayNode:
            isVisible = lineDisplayNode.GetVisibility()  # Check current visibility
            print(f"[Debug] Updating reference line for {side}. Visibility: {'ON' if isVisible else 'OFF'}.")
            toggleReferenceLine("Left" if side == "L" else "Right", isVisible)


def toggleReferenceLine(side, state):
    """
    Update the positions of reference lines (horizontal and vertical) for the specified side ('Left' or 'Right').
    Hide lines from the opposite side if they exist.
    The state parameter controls the visibility of the lines.
    """
    global Head_Transform_L, Head_Transform_R, L_Tr_Transform, R_Tr_Transform

    # Determine the appropriate transforms and names based on the side
    headTransform = Head_Transform_L if side == "Left" else Head_Transform_R
    trochanterTransform = L_Tr_Transform if side == "Left" else R_Tr_Transform
    lineNameHorizontal = f"Upper Reference ({side})"
    lineNameVertical = f" H-Tr Distance \n "

    # Opposite side details
    oppositeSide = "Right" if side == "Left" else "Left"
    oppositeLineHorizontal = f"Upper Reference ({oppositeSide})"
    oppositeLineVertical = f" H-Tr Distance:\n )"

    # Hide lines from the opposite side
    for lineName in [oppositeLineHorizontal, oppositeLineVertical]:
        lineNode = slicer.mrmlScene.GetFirstNodeByName(lineName)
        if lineNode:
            displayNode = lineNode.GetDisplayNode()
            if displayNode:
                displayNode.SetVisibility(False)
                print(f"[Debug] Hidden line '{lineName}' from the opposite side.")

    # Ensure required transforms are available
    if not headTransform:
        slicer.util.errorDisplay(f"Head for side '{side}' is missing. Please add the head first.")
        return
    if not trochanterTransform:
        slicer.util.errorDisplay(f"Trochanter for side '{side}' is missing. Please add it first.")
        return

    # Get head position
    headMatrix = vtk.vtkMatrix4x4()
    headTransform.GetMatrixTransformToParent(headMatrix)
    headPosition = [
        headMatrix.GetElement(0, 3),  # X
        headMatrix.GetElement(1, 3),  # Y
        headMatrix.GetElement(2, 3)   # Z
    ]

    # Get trochanter position
    trochanterMatrix = vtk.vtkMatrix4x4()
    trochanterTransform.GetMatrixTransformToParent(trochanterMatrix)
    trochanterX = trochanterMatrix.GetElement(0, 3)
    trochanterZ = trochanterMatrix.GetElement(2, 3)

    # Define points for horizontal and vertical lines
    startPointHorizontal = headPosition
    endPointHorizontal = [trochanterX, headPosition[1], headPosition[2]]

    startPointVertical = endPointHorizontal
    endPointVertical = [trochanterX, headPosition[1], trochanterZ]

    # Create or update the horizontal line
    lineNodeHorizontal = slicer.mrmlScene.GetFirstNodeByName(lineNameHorizontal)
    if not lineNodeHorizontal:
        lineNodeHorizontal = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", lineNameHorizontal)
        lineNodeHorizontal.AddControlPoint(startPointHorizontal, "head center")
        lineNodeHorizontal.AddControlPoint(endPointHorizontal)
    else:
        lineNodeHorizontal.SetNthControlPointPosition(0, startPointHorizontal)
        lineNodeHorizontal.SetNthControlPointPosition(1, endPointHorizontal)

    # Set horizontal line appearance: thin and gray
    horizontalDisplayNode = lineNodeHorizontal.GetDisplayNode()
    if horizontalDisplayNode:
        horizontalDisplayNode.SetLineThickness(0.1)  # Thin line
        horizontalDisplayNode.SetColor(0.5, 0.5, 0.5)  # Gray color
        horizontalDisplayNode.SetTextScale(0)  # Hide text
        horizontalDisplayNode.SetVisibility(state)

    # Create or update the vertical line
    lineNodeVertical = slicer.mrmlScene.GetFirstNodeByName(lineNameVertical)
    if not lineNodeVertical:
        lineNodeVertical = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", lineNameVertical)
        lineNodeVertical.AddControlPoint(startPointVertical, "trochanter position")
        lineNodeVertical.AddControlPoint(endPointVertical)
    else:
        lineNodeVertical.SetNthControlPointPosition(0, startPointVertical)
        lineNodeVertical.SetNthControlPointPosition(1, endPointVertical)

    # Set vertical line appearance: thicker and labeled
    verticalDisplayNode = lineNodeVertical.GetDisplayNode()
    if verticalDisplayNode:
        verticalDisplayNode.SetLineThickness(0.3)  # Thicker line
        verticalDisplayNode.SetSelectedColor(1, 1, 0)  # Yellow color
        verticalDisplayNode.SetTextScale(3.0)  # Label text scale
        verticalDisplayNode.SetVisibility(state)

    # Update visibility based on checkbox state
    for lineNode in [lineNodeHorizontal, lineNodeVertical]:
        displayNode = lineNode.GetDisplayNode()
        if displayNode:
            displayNode.SetVisibility(state)
            print(f"[Debug] {'Displayed' if state else 'Hidden'} line '{lineNode.GetName()}'.")


def calculate_corrected_angles():
    """
    Calculate the angles between the Ischial line and vectors to Left and Right Trochanters.

    Returns:
        tuple: (angle_end, angle_start), angles in degrees.
    """
    global Ischial_Line_Node, L_Tr_Node, R_Tr_Node

    # Sprawdzanie obecności wymaganych węzłów
    if not Ischial_Line_Node or not L_Tr_Node or not R_Tr_Node:
        print("[Warning] Required nodes are missing.")
        return None, None

    # Pobierz punkty linii Ischial
    line_start = [0.0, 0.0, 0.0]
    line_end = [0.0, 0.0, 0.0]
    Ischial_Line_Node.GetNthControlPointPositionWorld(0, line_start)
    Ischial_Line_Node.GetNthControlPointPositionWorld(1, line_end)

    # Pobierz pozycje trochanterów
    left_trochanter_pos = get_world_position(L_Tr_Node)
    right_trochanter_pos = get_world_position(R_Tr_Node)

    if left_trochanter_pos is None or right_trochanter_pos is None:
        print("[Warning] Trochanter positions are missing.")
        return None, None

    # Oblicz wektory
    vector_start_to_left = np.array(left_trochanter_pos) - np.array(line_start)
    vector_start_to_end = np.array(line_end) - np.array(line_start)
    vector_end_to_right = np.array(right_trochanter_pos) - np.array(line_end)
    vector_end_to_start = np.array(line_start) - np.array(line_end)

    # Normalizacja wektorów
    vector_start_to_left /= np.linalg.norm(vector_start_to_left)
    vector_start_to_end /= np.linalg.norm(vector_start_to_end)
    vector_end_to_right /= np.linalg.norm(vector_end_to_right)
    vector_end_to_start /= np.linalg.norm(vector_end_to_start)

    # Oblicz kąty
    angle_start_rad = np.arccos(np.clip(np.dot(vector_start_to_left, vector_start_to_end), -1.0, 1.0))
    angle_end_rad = np.arccos(np.clip(np.dot(vector_end_to_right, vector_end_to_start), -1.0, 1.0))

    # Konwersja do stopni
    angle_start = np.degrees(angle_start_rad)
    angle_end = np.degrees(angle_end_rad)

    # Debug: Wyświetl wartości kątów
    print(f"[Debug] Angle at start (line start): {angle_start:.2f}°")
    print(f"[Debug] Angle at end (line end): {angle_end:.2f}°")

    return angle_end, angle_start


def calculate_and_display_corrected_positions():
    """
    Calculate vertical shifts and display corrected positions of the Trochanters based on the selected reference side.
    """
    global L_Tr_Node, R_Tr_Node, L_Cup_Transform, R_Cup_Transform, leftReferenceButton, rightReferenceButton
    global left_vertical_shift, right_vertical_shift

    # Determine the selected side from the radiobuttons
    if leftReferenceButton.isChecked():
        side = "L"
    elif rightReferenceButton.isChecked():
        side = "R"
    else:
        slicer.util.errorDisplay("No side selected. Please select a reference side (Left or Right).")
        return

    # Debugging output for side selection
    print(f"[Debug] calculate_and_display_corrected_positions called for side: {side}")

    # Define variables based on the side
    if side == "L":
        segment_name = "Corrected Left Trochanter"
        trochanter_node = L_Tr_Node
        cup_transform = L_Cup_Transform
        vertical_shift = left_vertical_shift if 'left_vertical_shift' in globals() else 0
    elif side == "R":
        segment_name = "Corrected Right Trochanter"
        trochanter_node = R_Tr_Node
        cup_transform = R_Cup_Transform
        vertical_shift = right_vertical_shift if 'right_vertical_shift' in globals() else 0

    # Check if the corrected trochanter is already displayed
    fiducial_node = slicer.mrmlScene.GetFirstNodeByName(segment_name)
    if fiducial_node:
        # If the corrected trochanter exists, hide it
        slicer.mrmlScene.RemoveNode(fiducial_node)
        print(f"[Debug] Removed existing fiducial for {segment_name}")
        return

    # Check if required nodes are available
    if not trochanter_node or not cup_transform:
        print(f"[Info] Nodes for side {side} are not yet available. Skipping corrected positions display.")
        return

    # Get Trochanter and Cup positions
    trochanter_pos = get_world_position(trochanter_node)
    cup_center = get_cup_center(cup_transform)
    if trochanter_pos is None or cup_center is None:
        print(f"[Warning] {side} Trochanter or Cup positions could not be retrieved.")
        return

    print(f"[Debug] Trochanter Position ({side}): {np.array(trochanter_pos)}")
    print(f"[Debug] Cup Center ({side}): {np.array(cup_center)}")

    # Calculate angles and angle difference
    angle_end, angle_start = calculate_corrected_angles()
    angle_difference = angle_start - angle_end
    print(f"[Debug] Angle Difference: {angle_difference:.2f}°")

    # Calculate corrected position and vertical shift
    corrected_pos, vertical_shift = calculate_corrected_position_and_vertical_shift(
        trochanter_pos=np.array(trochanter_pos),
        cup_center=np.array(cup_center),
        angle_difference=angle_difference
    )

    print(f"[Debug] Corrected Trochanter Position ({side}): {corrected_pos}, Vertical Shift: {vertical_shift:.2f} mm")

    # Display corrected position as fiducial
    display_fiducial_point(corrected_pos, segment_name)

    # Update the head geometry and position
    print(f"[Debug] Calling updateHead for side: {side}")
    updateHead(side)



def remove_fiducials_by_name(fiducial_name):
    """
    Remove all fiducials with the specified name.
    """
    nodes_to_remove = slicer.mrmlScene.GetNodesByClass("vtkMRMLMarkupsFiducialNode")
    for i in range(nodes_to_remove.GetNumberOfItems()):
        fiducial_node = nodes_to_remove.GetItemAsObject(i)
        if fiducial_node.GetName() == fiducial_name:
            slicer.mrmlScene.RemoveNode(fiducial_node)

def calculate_corrected_position_and_vertical_shift(trochanter_pos, cup_center, angle_difference):
    """
    Calculate the corrected position of a Trochanter point and the vertical (Z-axis) shift after rotation in the XZ plane.

    Args:
        trochanter_pos (np.array): Coordinates of the Trochanter [X, Y, Z].
        cup_center (np.array): Coordinates of the cup center [X, Y, Z].
        angle_difference (float): Difference of angles (degrees) for the rotation.

    Returns:
        tuple: Corrected position [X, Y, Z] and vertical shift (float) in millimeters.
    """
    # Convert the angle difference to radians
    angle_radians = np.radians(angle_difference)
    
    # Calculate the radius vector (relative position of the Trochanter to the center)
    radius_vector = trochanter_pos - cup_center
    radius_vector_xz = np.array([radius_vector[0], 0, radius_vector[2]])  # Ignore Y component
    
    # Rotation matrix for counter-clockwise rotation around Y-axis (in XZ plane)
    rotation_matrix = np.array([
        [np.cos(angle_radians), 0, np.sin(angle_radians)],
        [0,                     1, 0                    ],
        [-np.sin(angle_radians), 0, np.cos(angle_radians)]
    ])
    
    # Rotate the Trochanter point around the cup center
    rotated_local_point = np.dot(rotation_matrix, radius_vector_xz)
    
    # Add the rotated local point back to the cup center to get world coordinates
    corrected_position = rotated_local_point + cup_center
    
    # Calculate the vertical shift in Z-axis
    vertical_shift = rotated_local_point[2] - radius_vector_xz[2]
    
    return corrected_position, vertical_shift

def get_world_position(node):
    """
    Get the world position of a node.
    """
    if not node:
        return None
    
    position = [0.0, 0.0, 0.0]
    transform_node = node.GetParentTransformNode()
    
    if transform_node:
        # Transform local point to world
        local_point = np.array([0, 0, 0, 1])  # Local origin
        world_matrix = vtk.vtkMatrix4x4()
        transform_node.GetMatrixTransformToWorld(world_matrix)
        
        world_point = np.dot(
            np.array([[world_matrix.GetElement(i, j) for j in range(4)] for i in range(4)]),
            local_point
        )
        position = world_point[:3]
    else:
        # Use raw origin if no transform
        node.GetOrigin(position)
    
    return position


def get_cup_center(cup_transform):
    """
    Get the center of a cup (left or right) from its transform.
    """
    if not cup_transform:
        return None
    
    transform_matrix = vtk.vtkMatrix4x4()
    cup_transform.GetMatrixTransformToWorld(transform_matrix)
    return [
        transform_matrix.GetElement(0, 3),
        transform_matrix.GetElement(1, 3),
        transform_matrix.GetElement(2, 3),
    ]


def display_fiducial_point(position, name):
    """
    Display a fiducial point in the 3D scene using AddControlPoint.

    Args:
        position (np.array): Coordinates of the point [X, Y, Z].
        name (str): Name of the fiducial point.
    """
    fiducial_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", name)
    fiducial_node.AddControlPoint(position[0], position[1], position[2])

# Funkcja obsługująca dolną linię referencyjną
def toggleLowerReferenceLine(side):
    """
    Toggle visibility or update a lower reference line named 'Lower Reference' for the specified side ('Left' or 'Right').
    The line is tilted at a 45-degree angle in the X-Z plane.
    """
    global Head_Transform_L, Head_Transform_R, referenceLineOffset, horizontalLineLength

    # Select the appropriate transform based on the side
    headTransform = Head_Transform_L if side == "Left" else Head_Transform_R
    lineName = f"Lower Reference ({side})"

    if not headTransform:
        slicer.util.errorDisplay(f"Head for side '{side}' is missing. Please add the head first.")
        return

    # Get the head position
    matrix = vtk.vtkMatrix4x4()
    headTransform.GetMatrixTransformToParent(matrix)
    headPosition = [
        matrix.GetElement(0, 3),  # X
        matrix.GetElement(1, 3),  # Y
        matrix.GetElement(2, 3) - referenceLineOffset  # Z, adjusted by referenceLineOffset
    ]

    # Compute the start and end points for the tilted line
    if side == "Left":
        startPoint = headPosition  # Starting point: head center
        endPoint = [
            headPosition[0] - horizontalLineLength,  # Move horizontally in the X direction
            headPosition[1],                         # Keep the same Y-coordinate
            headPosition[2] + horizontalLineLength   # Tilt upwards in the Z direction at 45 degrees
        ]
    else:
        startPoint = [
            headPosition[0] + horizontalLineLength,  # Move horizontally in the X direction
            headPosition[1],                         # Keep the same Y-coordinate
            headPosition[2] + horizontalLineLength   # Tilt upwards in the Z direction at 45 degrees
        ]
        endPoint = headPosition  # Ending point: head center

    # Check if the line already exists
    lineNode = slicer.mrmlScene.GetFirstNodeByName(lineName)
    if lineNode:
        # Update the positions of the control points
        lineNode.SetNthControlPointPosition(0, startPoint)
        lineNode.SetNthControlPointPosition(1, endPoint)

        # Toggle the visibility of the line
        displayNode = lineNode.GetDisplayNode()
        if displayNode:
            displayNode.SetVisibility(not displayNode.GetVisibility())
            print(f"[Debug] Updated and toggled visibility for '{lineName}'.")
        return

    # Create a new reference line
    lineNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", lineName)
    lineNode.AddControlPoint(startPoint, "reamer top")  # Add text: reamer top
    lineNode.AddControlPoint(endPoint)  # End point

    # Customize visual properties of the line
    displayNode = lineNode.GetDisplayNode()
    if displayNode:
        displayNode.SetSelectedColor(0, 1, 0)  # Green color
        displayNode.SetLineThickness(0.25)     # Line thickness 0.25
        displayNode.SetVisibility(True)
        displayNode.SetHandlesInteractive(False)  # Disable interactivity
        displayNode.SetOpacity(0.8)

        # Set text scale
        displayNode.SetTextScale(2.0)  # Text scale

        print(f"[Debug] Lower Reference line '{lineName}' created: Start {startPoint}, End {endPoint}.")

def displayDistanceZ(side, headTransform, trTransform, correctedShift=0):
    """
    Oblicza i wyświetla odległość w osi Z między środkiem głowy a punktem TR dla danej strony.
    Dodatkowo wyświetla skorygowaną odległość, jeśli podano przesunięcie.
    """
    if not headTransform or not trTransform:
        slicer.util.errorDisplay(f"Brak transformacji dla strony {side}. Nie można obliczyć odległości.")
        return

    # Ukryj etykietę przeciwnej strony, jeśli istnieje
    otherSide = "R" if side == "L" else "L"
    otherLabelNode = slicer.mrmlScene.GetFirstNodeByName(f"Distance Z Display {otherSide}")
    if otherLabelNode:
        otherDisplayNode = otherLabelNode.GetDisplayNode()
        if otherDisplayNode:
            otherDisplayNode.SetVisibility(False)

    def getTransformCenterZ(transformNode):
        matrix = vtk.vtkMatrix4x4()
        transformNode.GetMatrixTransformToParent(matrix)
        return matrix.GetElement(2, 3)

    headZ = getTransformCenterZ(headTransform)
    trZ = getTransformCenterZ(trTransform)
    if headZ is None or trZ is None:
        slicer.util.errorDisplay(f"Nie można odczytać współrzędnych Z dla strony {side}.")
        return

    distanceZ = headZ - trZ

    # Obliczenie skorygowanej odległości, jeśli podano przesunięcie
    correctedDistanceZ = None
    if correctedShift:
        correctedHeadZ = headZ + correctedShift
        correctedDistanceZ = correctedHeadZ - trZ

    # Pobierz lub utwórz węzeł markupu do wyświetlania odległości
    nodeName = f"Distance Z Display {side}"
    fiducialNode = slicer.mrmlScene.GetFirstNodeByName(nodeName)
    if not fiducialNode:
        fiducialNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", nodeName)
        fiducialNode.AddControlPoint([0, 0, 0])  # Tymczasowa pozycja

    # Ustawienie pozycji etykiety w środku volumenu
    try:
        volumeNode = slicer.util.getNode("vtkMRMLScalarVolumeNode*")
        bounds = [0.0] * 6
        volumeNode.GetRASBounds(bounds)
        volumeCenter = [
            (bounds[0] + bounds[1]) / 2.0,
            (bounds[2] + bounds[3]) / 2.0,
            bounds[4] + 10,
        ]
        fiducialNode.SetNthControlPointPosition(0, volumeCenter)
    except:
        fiducialNode.SetNthControlPointPosition(0, [0, 0, 0])

    # Ustawienie tekstu etykiety z odległościami
    labelText = f"Head {side} - TR Distance Z: {distanceZ:.2f} mm"
    if correctedDistanceZ is not None:
        labelText += f"\nCorrected: {correctedDistanceZ:.2f} mm"
    fiducialNode.SetNthControlPointLabel(0, labelText)

    displayNode = fiducialNode.GetDisplayNode()
    if not displayNode:
        displayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsDisplayNode")
        fiducialNode.SetAndObserveDisplayNodeID(displayNode.GetID())
    displayNode.SetGlyphScale(0)
    displayNode.SetTextScale(5)
    displayNode.SetColor(0, 1, 0)
    displayNode.SetVisibility(True)

    print(f"[Debug] {labelText}")

def testLabelPosition(segmentationNode, segmentID, labelName, textOffset=5, textScale=3.0):
    """
    Test adding a label without any dynamic transform, just based on global segment center.
    """
    if segmentationNode is None:
        print(f"[Test Error] Segmentation node is None for label '{labelName}'. Exiting.")
        return

    # Calculate the global center of the segment
    globalCenter = getSegmentCenter(segmentationNode, segmentID)
    if globalCenter == [0.0, 0.0, 0.0]:
        print(f"[Test Error] Segment '{segmentID}' center could not be calculated.")
        return
    print(f"[Test] Global center for segment '{segmentID}': {globalCenter}")

    # Adjust position for the label
    labelPosition = [
        globalCenter[0],  # X
        globalCenter[1],  # Y
        globalCenter[2] + textOffset,  # Z
    ]
    print(f"[Test] Calculated label position: {labelPosition}")

    # Add or update label in the scene
    fiducialNode = slicer.mrmlScene.GetFirstNodeByName(labelName)
    if fiducialNode:
        fiducialNode.SetNthControlPointPosition(0, *labelPosition)
        print(f"[Test] Updated label '{labelName}' position: {labelPosition}")
    else:
        fiducialNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", labelName)
        fiducialNode.AddControlPoint(labelPosition, segmentationNode.GetName())
        print(f"[Test] Created label '{labelName}' at position: {labelPosition}")

    # Set display properties
    displayNode = fiducialNode.GetDisplayNode()
    if displayNode:
        displayNode.SetGlyphScale(0)  # Hide glyph marker
        displayNode.SetTextScale(textScale)
        displayNode.SetColor(1, 0, 0)  # Set color to red for testing
        displayNode.SetVisibility(True)
        displayNode.SetVisibility3D(True)
    else:
        print(f"[Test Error] No display node for label '{labelName}'.")

def transformLocalToGlobal(localPoint, transformNode):
    """
    Transform a local point to global coordinates using the transform node.
    
    :param localPoint: A list [x, y, z] representing the local point.
    :param transformNode: The transform node to apply.
    :return: A list [x, y, z] representing the global coordinates.
    """
    if not transformNode:
        print(f"[Debug] No transform node provided. Returning local point: {localPoint}")
        return localPoint  # No transform applied

    # Create a 4x4 transformation matrix
    transformMatrix = vtk.vtkMatrix4x4()
    transformNode.GetMatrixTransformToWorld(transformMatrix)

    # Transform the local point to global coordinates
    localPoint.append(1.0)  # Add homogeneous coordinate for transformation
    globalPoint = [0.0, 0.0, 0.0, 1.0]
    transformMatrix.MultiplyPoint(localPoint, globalPoint)
    globalPoint = globalPoint[:3]  # Remove homogeneous coordinate
    print(f"[Debug] Transformed local point {localPoint[:3]} to global point {globalPoint} using transform '{transformNode.GetName()}'.")
    return globalPoint

def debugTransform(transformNode):
    """
    Debugging function to display the full transform matrix of a node.
    """
    if not transformNode:
        print("[Debug] Transform node is None.")
        return
    if not isinstance(transformNode, slicer.vtkMRMLTransformNode):
        print(f"[Debug] Node '{transformNode.GetName()}' is not a transform node.")
        return

    transformMatrix = vtk.vtkMatrix4x4()
    transformNode.GetMatrixTransformToWorld(transformMatrix)
    print(f"[Debug] Transform matrix for node '{transformNode.GetName()}':")
    for i in range(4):
        row = [transformMatrix.GetElement(i, j) for j in range(4)]
        print(f"    {row}")

# Globalna flaga do śledzenia stanu manipulatorów
manipulatorsEnabled = False

def toggleManipulatorsForVisibleElements(interactorStyle, event):
    """
    Toggle manipulators for visible Tr and Cup elements.
    """
    global manipulatorsEnabled, L_Cup_Transform, R_Cup_Transform, L_Tr_Transform, R_Tr_Transform
    global L_Cup_Checkbox, R_Cup_Checkbox, L_Tr_Checkbox, R_Tr_Checkbox

    print("[Manipulators] Double-click event detected.")

    # Lista transformów do obsługi
    transforms = [L_Cup_Transform, R_Cup_Transform, L_Tr_Transform, R_Tr_Transform]
    checkboxes = [L_Cup_Checkbox, R_Cup_Checkbox, L_Tr_Checkbox, R_Tr_Checkbox]
    visibleTransforms = []

    # Funkcja pomocnicza: Sprawdza widoczność powiązanej segmentacji
    def isVisible(transformNode):
        if not transformNode:
            return False
        for segmentationNode in slicer.util.getNodesByClass("vtkMRMLSegmentationNode"):
            if segmentationNode.GetTransformNodeID() == transformNode.GetID():
                displayNode = segmentationNode.GetDisplayNode()
                if displayNode and displayNode.GetVisibility():
                    return True
        return False

    # Funkcja pomocnicza: Tworzy DisplayNode, jeśli nie istnieje
    def ensureDisplayNode(transformNode):
        if not transformNode.GetDisplayNode():
            transformNode.CreateDefaultDisplayNodes()
        return transformNode.GetDisplayNode()

    # Znalezienie transformacji z widocznymi segmentacjami
    for transform in transforms:
        if isVisible(transform):
            visibleTransforms.append(transform)

    if not visibleTransforms:
        print("[Manipulators] No visible elements to process.")
        return

    # Przełączanie manipulatorów **tylko dla transformacji**
    if manipulatorsEnabled:
        print("[Manipulators] Disabling manipulators for visible elements.")
        for transform, checkbox in zip(visibleTransforms, checkboxes):
            displayNode = ensureDisplayNode(transform)
            if displayNode:
                displayNode.SetEditorVisibility(False)
                if checkbox:
                    checkbox.setChecked(False)
    else:
        print("[Manipulators] Enabling manipulators for visible elements.")
        for transform, checkbox in zip(visibleTransforms, checkboxes):
            displayNode = ensureDisplayNode(transform)
            if displayNode:
                displayNode.SetEditorVisibility(True)
                if checkbox:
                    checkbox.setChecked(True)

    # Aktualizacja globalnego stanu manipulatorów
    manipulatorsEnabled = not manipulatorsEnabled
    print(f"[Manipulators] Manipulators are now {'enabled' if manipulatorsEnabled else 'disabled'}.")

def registerDoubleClickEventFor2DViews():
    """
    Register double-click events on all 2D slice views to toggle manipulators.
    """
    layoutManager = slicer.app.layoutManager()
    for sliceViewName in layoutManager.sliceViewNames():
        sliceWidget = layoutManager.sliceWidget(sliceViewName)
        interactor = sliceWidget.sliceView().interactor()

        # Usunięcie wcześniejszych rejestracji, jeśli były
        interactor.RemoveObservers("LeftButtonDoubleClickEvent")
        
        # Dodanie nowego obserwatora
        interactor.AddObserver("LeftButtonDoubleClickEvent", toggleManipulatorsForVisibleElements)
        print(f"[Debug] Double-click event registered for {sliceViewName}.")

# Rejestracja zdarzenia podwójnego kliknięcia dla wszystkich widoków 2D
registerDoubleClickEventFor2DViews()

def refresh2DViews():
    """
    Force refresh of 2D slice views to ensure visibility of segmentation changes.
    """
    layoutManager = slicer.app.layoutManager()
    for sliceViewName in layoutManager.sliceViewNames():
        sliceWidget = layoutManager.sliceWidget(sliceViewName)
        sliceWidget.sliceLogic().FitSliceToAll()

refresh2DViews()

def isUpperReferenceLineVisible():
    """
    Check if the upper reference line for the selected side is visible.
    """
    side = "Left" if leftReferenceButton.isChecked() else "Right"
    lineName = f"Upper Reference ({side})"
    lineNode = slicer.mrmlScene.GetFirstNodeByName(lineName)
    if lineNode and lineNode.GetDisplayNode():
        return lineNode.GetDisplayNode().GetVisibility()
    return False


def createCupToolbar():
    """
    Create a toolbar for managing Cups, Trs, Heads, transforms, STL segmentations,
    including reset, X-ray, interaction with transforms, and DICOM scaling.
    """
    global distanceLabel, currentSegmentationNode, currentSegmentID, XRay_Transform_Node, xrayScaleSpinBox
    global actualSizeInput, measuredSizeInput
    global L_Tr_Checkbox, R_Tr_Checkbox, L_Cup_Checkbox, R_Cup_Checkbox, Ischial_Checkbox
    global referenceLineButton, leftReferenceButton, rightReferenceButton
    global cupSizeSpinBoxL, cupSizeSpinBoxR
    
    def applyModernToolbarStyles(toolbar):
        """Apply modern, colorful styling with better text visibility"""
        toolbar.setStyleSheet("""
            QToolBar {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2c3e50, stop:1 #34495e);
                border: 2px solid #3498db;
                border-radius: 10px;
                padding: 8px;
                spacing: 4px;
                color: white;
            }
            
            QScrollArea {
                background: transparent;
                border: none;
                margin: 0px;
                padding: 0px;
            }
            
            QScrollBar:vertical {
                background: #34495e;
                width: 12px;
                border-radius: 6px;
                margin: 0px;
            }
            
            QScrollBar::handle:vertical {
                background: #3498db;
                border-radius: 6px;
                min-height: 20px;
            }
            
            QScrollBar::handle:vertical:hover {
                background: #5dade2;
            }
            
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
            
            QToolBar QWidget {
                margin: 2px;
                color: white;
            }
            
            /* DICOM Import Group - Blue Theme */
            QPushButton[group="dicom"] {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3498db, stop:1 #2980b9);
                border: 2px solid #2980b9;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 11px;
                font-weight: bold;
                color: white;
                min-height: 24px;
            }
            
            QPushButton[group="dicom"]:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #5dade2, stop:1 #3498db);
                border-color: #85c1e9;
            }
            
            /* Anatomy Group - Green Theme */
            QPushButton[group="anatomy"] {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #27ae60, stop:1 #229954);
                border: 2px solid #229954;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 11px;
                font-weight: bold;
                color: white;
                min-height: 24px;
                min-width: 80px;
                max-width: 80px;
            }
            
            QPushButton[group="anatomy"]:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #58d68d, stop:1 #27ae60);
            }
            
            /* Control Group - Orange Theme */
            QPushButton[group="control"] {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #e67e22, stop:1 #d35400);
                border: 2px solid #d35400;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 11px;
                font-weight: bold;
                color: white;
                min-height: 24px;
            }
            
            QPushButton[group="control"]:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f39c12, stop:1 #e67e22);
            }
            
            /* STL Group - Purple Theme */
            QPushButton[group="stl"] {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #8e44ad, stop:1 #7d3c98);
                border: 2px solid #7d3c98;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 11px;
                font-weight: bold;
                color: white;
                min-height: 24px;
            }
            
            QPushButton[group="stl"]:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #a569bd, stop:1 #8e44ad);
            }
            
            /* Action Group - Red/Gray Theme */
            QPushButton[group="action"] {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #e74c3c, stop:1 #c0392b);
                border: 2px solid #c0392b;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 11px;
                font-weight: bold;
                color: white;
                min-height: 24px;
            }
            
            QPushButton[group="action"]:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ec7063, stop:1 #e74c3c);
            }
            
            QSpinBox, QComboBox, QLineEdit {
                background: #34495e;
                border: 2px solid #7f8c8d;
                border-radius: 4px;
                padding: 4px 6px;
                font-size: 11px;
                font-weight: bold;
                color: white;
                min-height: 20px;
                max-width: 100px;
            }
            
            QSpinBox:focus, QComboBox:focus, QLineEdit:focus {
                border-color: #3498db;
                background: #2c3e50;
            }
            
            QSpinBox::up-button {
                background: #3498db;
                border: 1px solid #2980b9;
                border-radius: 2px;
                width: 16px;
            }
            
            QSpinBox::up-button:hover {
                background: #5dade2;
            }
            
            QSpinBox::up-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-bottom: 6px solid white;
                width: 0px;
                height: 0px;
            }
            
            QSpinBox::down-button {
                background: #3498db;
                border: 1px solid #2980b9;
                border-radius: 2px;
                width: 16px;
            }
            
            QSpinBox::down-button:hover {
                background: #5dade2;
            }
            
            QSpinBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid white;
                width: 0px;
                height: 0px;
            }
            
            QComboBox::drop-down {
                background: #3498db;
                border: 1px solid #2980b9;
                border-radius: 2px;
                width: 20px;
            }
            
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid white;
                width: 0px;
                height: 0px;
            }
            
            QLabel {
                font-size: 11px;
                font-weight: bold;
                color: white;
                margin: 2px 4px;
                background: transparent;
            }
            
            QCheckBox {
                font-size: 11px;
                font-weight: bold;
                color: white;
                spacing: 4px;
                background: transparent;
            }
            
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 2px solid #bdc3c7;
                border-radius: 3px;
                background: white;
            }
            
            QCheckBox::indicator:checked {
                background: #27ae60;
                border-color: #27ae60;
            }
            
            QRadioButton {
                font-size: 11px;
                font-weight: bold;
                color: white;
                spacing: 4px;
                background: transparent;
            }
            
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                background: white;
            }
            
            QRadioButton::indicator:checked {
                background: #3498db;
                border-color: #3498db;
            }
        """)

    def createCompactContainer(widgets, spacing=3):
        """Create a compact container for widgets"""
        container = qt.QWidget()
        layout = qt.QHBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(spacing)
        
        for widget in widgets:
            layout.addWidget(widget)
        
        return container

    def addVerticalSpacer(layout, height=15):
        """Add a colorful vertical spacer to layout"""
        spacer = qt.QWidget()
        spacer.setFixedHeight(height)
        spacer.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #3498db, stop:0.5 #e74c3c, stop:1 #f39c12);
            margin: 4px 8px;
            border-radius: 2px;
        """)
        layout.addWidget(spacer)

    def readFirstInstruction():
        """Show first instruction dialog"""
        msg = qt.QMessageBox()
        msg.setWindowTitle("Easy Hip Planner - Instructions")
        msg.setTextFormat(qt.Qt.RichText)
        msg.setText("""
        <h3><b>🦴 Easy Hip Planner - Instructions</b></h3>
        
        <h4><b>⚠️ NOT FOR MEDICAL USE!!!</b></h4>
        
        <h4><b>🚀 QUICK and EASY START :)</b></h4>
        
        <h4><b>Basic use:</b></h4>
        <p><b>1.</b> Import X-ray DICOM image (if problems occur, use X-ray exported from RadiantViewer)<br>
        <b>2.</b> Press green "Show all" button<br>
        <b>3.</b> Place Cups in hip joint sockets according to L/R side<br>
        <b>4.</b> Place Tr on trochanter peaks (initially recommend greater ones)<br>
        <b>5.</b> Adjust Ischial line position to pelvis inclination (e.g., according to ischial tuberosities position)<br>
        <b>6.</b> Select operated side and click "Show Head"<br>
        <b>7.</b> Click "Toggle Upper Reference Line"<br>
        <b>8.</b> <b>IMPORTANT:</b> The result "Head-Tr distance" shows where the endoprosthesis head should be relative to your chosen point (e.g., greater trochanter peak), so the leg is equal to the other one (its trochanter).<br>
        <b>9.</b> Enjoy! :)</p>
        
        <h4><b>📝 Notes:</b></h4>
        <p>• Intraoperatively for orientation, you can use e.g., K-wire, aiming it through the trochanter peak to the endoprosthesis head to assess their relative position<br>
        • Don't worry that the head is not in the green socket - that's not the point, the socket is only a reference point<br>
        • If after scene reset the image turns into a strip, click the "X-ray mode" button to restore proper view</p>
        • Advanced mode tutorial with stem adding - coming soon! </p>
                    
        <p><i>💡 Double-click in 2D view enables/disables manipulators</i></p>
        <h4><b>📧 Contact:</b></h4>
        <p><b>Email:</b> przemek.czuma@gmail.com<br>
        <b>Website:</b> <a href="https://inteligencja.org.pl">inteligencja.org.pl</a><br>
        <b>LinkedIn:</b> <a href="https://www.linkedin.com/in/inteligencjawzdrowiu/">linkedin.com/in/inteligencjawzdrowiu</a></p>
    """)        
        msg.setStandardButtons(qt.QMessageBox.Ok)
        msg.exec_()


    # Remove existing toolbar
    existingToolbars = [toolbar for toolbar in slicer.util.mainWindow().findChildren(qt.QToolBar) if toolbar.windowTitle == "Cup Tool"]
    for toolbar in existingToolbars:
        slicer.util.mainWindow().removeToolBar(toolbar)

    # Create new toolbar
    toolbar = qt.QToolBar("Cup Tool")
    slicer.util.mainWindow().addToolBar(qt.Qt.LeftToolBarArea, toolbar)

    if not toolbar:
        print("[Debug] Failed to create toolbar.")
        return
    print("[Debug] Toolbar 'Cup Tool' created.")

    # Create scroll area for the toolbar content
    scrollArea = qt.QScrollArea()
    scrollArea.setWidgetResizable(True)
    scrollArea.setVerticalScrollBarPolicy(qt.Qt.ScrollBarAsNeeded)
    scrollArea.setHorizontalScrollBarPolicy(qt.Qt.ScrollBarAlwaysOff)
    
    # Create main content widget
    contentWidget = qt.QWidget()
    contentLayout = qt.QVBoxLayout(contentWidget)
    contentLayout.setContentsMargins(5, 5, 5, 5)
    contentLayout.setSpacing(3)
    
    # Title with better styling and Read First button
    titleContainer = qt.QWidget()
    titleLayout = qt.QHBoxLayout(titleContainer)
    titleLayout.setContentsMargins(0, 0, 0, 0)
    titleLayout.setSpacing(5)
    
    titleLabel = qt.QLabel("🦴 Easy Hip Planner")
    titleLabel.setStyleSheet("""
        color: #f39c12; 
        font-size: 16px; 
        font-weight: bold;
        padding: 8px 12px;
        margin: 4px;
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #34495e, stop:1 #2c3e50);
        border: 2px solid #f39c12;
        border-radius: 8px;
    """)
    
    readFirstButton = qt.QPushButton("📖 Read First")
    readFirstButton.setProperty("group", "control")
    readFirstButton.setMaximumWidth(100)
    readFirstButton.setToolTip("Pokaż instrukcję użycia\n• Krok po kroku przewodnik\n• Wyjaśnienie funkcji\n• Porady użytkowania")
    readFirstButton.clicked.connect(readFirstInstruction)
    
    titleLayout.addWidget(titleLabel)
    titleLayout.addWidget(readFirstButton)
    contentLayout.addWidget(titleContainer)

    # DICOM Import buttons - Blue theme with tooltips
    dicomContainer = createCompactContainer([
        qt.QPushButton("📁 X-ray"),
        qt.QPushButton("↻ Rotate"),
        qt.QPushButton("📁 CT")
    ])
    
    dicomButtons = dicomContainer.findChildren(qt.QPushButton)
    dicomTooltips = [
        "Import single X-ray DICOM file\n• Supports single slice selection\n• Automatically applies X-ray transform\n• Enables magnification scaling",
        "Rotate the green view by 90 degrees clockwise\n• Fixes image orientation\n• Useful for proper X-ray viewing",
        "Import CT DICOM folder\n• Loads entire CT series\n• Automatically selects most detailed series\n• Switches to Four-Up view"
    ]
    
    for i, (button, func, tooltip) in enumerate(zip(dicomButtons, 
                                                   [importDICOM, rotateGreenView, importCTFolder],
                                                   dicomTooltips)):
        button.setProperty("group", "dicom")
        button.clicked.connect(func)
        button.setToolTip(tooltip)
    
    contentLayout.addWidget(dicomContainer)
    
    addVerticalSpacer(contentLayout)
    
    def updateXRayScale(scalePercent):
        global XRay_Transform_Node, currentXRayScale
        try:
            volumeNode = slicer.util.getNode("vtkMRMLScalarVolumeNode*")
        except slicer.util.MRMLNodeNotFoundException:
            slicer.util.errorDisplay("No volume loaded. Please load a volume to update X-ray scaling.")
            return

        targetScaleFactor = scalePercent / 100.0
        correctionFactor = 1 / targetScaleFactor

        if not XRay_Transform_Node:
            XRay_Transform_Node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode", "X-Ray Transform")
            volumeNode.SetAndObserveTransformNodeID(XRay_Transform_Node.GetID())

        vtk_transform = vtk.vtkTransform()
        vtk_transform.Identity()
        vtk_transform.RotateX(90)
        vtk_transform.Scale(correctionFactor, correctionFactor, correctionFactor)

        transformMatrix = vtk.vtkMatrix4x4()
        vtk_transform.GetMatrix(transformMatrix)
        XRay_Transform_Node.SetMatrixTransformToParent(transformMatrix)

        reset2DAnd3DViews()
        slicer.app.processEvents()
        currentXRayScale = scalePercent
        print(f"[Debug] X-ray scaling updated to {scalePercent}%. Correction factor applied: {correctionFactor:.4f}")

    def onXrayScaleChanged(value):
        print(f"[Debug] X-ray scale spinbox changed to {value}. Resetting scene and applying transformation.")
        resetScene()
        updateXRayScale(value)

    # Scale controls with tooltip
    xrayScaleSpinBox = qt.QSpinBox()
    xrayScaleSpinBox.setRange(100, 125)
    xrayScaleSpinBox.setSingleStep(5)
    xrayScaleSpinBox.setValue(115)
    xrayScaleSpinBox.setKeyboardTracking(False)
    xrayScaleSpinBox.valueChanged.connect(onXrayScaleChanged)
    xrayScaleSpinBox.setToolTip("X-ray magnification scale (100-125%)\n• 115% is typical for X-rays\n• Adjusts image scaling automatically\n• Resets scene when changed")

    scaleContainer = createCompactContainer([
        qt.QLabel("Scale:"),
        xrayScaleSpinBox
    ])
    contentLayout.addWidget(scaleContainer)

    # Magnification controls with tooltips
    actualSizeInput = qt.QComboBox()
    actualSizeInput.setEditable(True)
    actualSizeInput.addItem("Marker: 30 mm", 30.0)
    actualSizeInput.addItem("Marker: 35 mm", 35.0)
    actualSizeInput.addItem("Marker: 40 mm", 40.0)
    actualSizeInput.addItem("Head: 22 mm", 22.0)
    actualSizeInput.addItem("Head: 28 mm", 28.0)
    actualSizeInput.addItem("Head: 32 mm", 32.0)
    actualSizeInput.addItem("Head: 36 mm", 36.0)
    actualSizeInput.addItem("1 dolar - 26.5 mm", 26.5)
    actualSizeInput.addItem("1 euro - 23.25 mm", 23.25)
    actualSizeInput.addItem("2 euro - 25.75 mm", 25.75)
    actualSizeInput.addItem("1 złoty - 23 mm", 23.0)
    actualSizeInput.addItem("5 zloty - 24 mm", 24.0)
    actualSizeInput.setCurrentIndex(1)
    actualSizeInput.setToolTip("Select the actual size of the reference object\n• Markers: 30-40mm\n• Femoral heads: 22-36mm\n• Coins for reference")
    
    measuredSizeInput = qt.QLineEdit()
    measuredSizeInput.setPlaceholderText("Click +Line to measure")
    measuredSizeInput.setToolTip("Shows the measured size from the image\n• Use the measurement line\n• Value will be auto-filled when using Apply button")
    
    # First line: Actual and Measured inputs
    magnificationContainer1 = createCompactContainer([
        qt.QLabel("Actual:"),
        actualSizeInput,
        qt.QLabel("Measured:"),
        measuredSizeInput
    ])
    contentLayout.addWidget(magnificationContainer1)

    # Second line: Apply and +Line buttons with tooltips
    applyButton = qt.QPushButton("Apply")
    applyButton.setProperty("group", "control")
    applyButton.clicked.connect(applyMagnification)
    applyButton.setToolTip("Calculate and apply magnification correction\n• Uses marker line measurement\n• Auto-fills measured size\n• Updates X-ray scale if within range (100-125%)")

    createNewLineButton = qt.QPushButton("+ Line")
    createNewLineButton.setProperty("group", "control")
    createNewLineButton.setToolTip("Create measurement line for magnification\n• Creates 'xray marker line'\n• Positioned 5cm from bottom center\n• Use to measure reference objects")

    def onCreateNewLine():
        try:
            print("[Debug] Create New Line button clicked.")
            addOrToggleLine(name="xray marker line")
            
            lineNode = slicer.mrmlScene.GetFirstNodeByName("xray marker line")
            if lineNode:
                displayNode = lineNode.GetDisplayNode()
                if displayNode:
                    displayNode.SetHandlesInteractive(False)
                    displayNode.SetLineThickness(0.1)
                    displayNode.SetTextScale(2.0)
                    displayNode.SetVisibility(True)

                    startPoint = [0.0, 0.0, 0.0]
                    endPoint = [0.0, 0.0, 0.0]
                    lineNode.GetNthControlPointPositionWorld(0, startPoint)
                    lineNode.GetNthControlPointPositionWorld(1, endPoint)
                    length = vtk.vtkMath.Distance2BetweenPoints(startPoint, endPoint) ** 0.5
                    lineNode.SetNthControlPointLabel(1, f"Length: {length:.2f} mm")

                    volumeNode = slicer.util.getNode("vtkMRMLScalarVolumeNode*")
                    bounds = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    volumeNode.GetRASBounds(bounds)
                    centerX = (bounds[0] + bounds[1]) / 2.0
                    bottomZ = bounds[4]
                    offsetZ = bottomZ + 50.0

                    startPoint = [centerX - 25.0, 0.0, offsetZ]
                    endPoint = [centerX + 25.0, 0.0, offsetZ]
                    lineNode.SetNthControlPointPosition(0, startPoint)
                    lineNode.SetNthControlPointPosition(1, endPoint)

                    print("[Debug] Line properties set: No handles, thin line, large text, length displayed, centered, 5 cm from bottom.")
            print("[Debug] Line added or toggled successfully.")
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to create or toggle line: {e}")
            print(f"[Error] Unexpected error in onCreateNewLine: {e}")

    createNewLineButton.clicked.connect(onCreateNewLine)

    magnificationContainer2 = createCompactContainer([
        applyButton,
        createNewLineButton
    ])
    contentLayout.addWidget(magnificationContainer2)

    addVerticalSpacer(contentLayout)

    # Control buttons with tooltips
    def onShowAll():
        elements = [
            (L_Tr_Node, lambda: toggleTrVisibilityOrAdd(10, "L"), canEnableLTrCheckbox, L_Tr_Checkbox),
            (R_Tr_Node, lambda: toggleTrVisibilityOrAdd(10, "R"), canEnableRTrCheckbox, R_Tr_Checkbox),
            (L_Cup_Node, lambda: toggleCupVisibilityOrAdd(50, "L"), canEnableLCupCheckbox, L_Cup_Checkbox),
            (R_Cup_Node, lambda: toggleCupVisibilityOrAdd(50, "R"), canEnableRCupCheckbox, R_Cup_Checkbox),
            (Ischial_Line_Node, lambda: addOrToggleLine("Ischial Line"), canEnableIschialCheckbox, Ischial_Checkbox),
        ]

        def isVisible(node):
            if node is None or node.GetDisplayNode() is None:
                return False
            return node.GetDisplayNode().GetVisibility()

        allVisible = all(isVisible(node) for node, _, _, _ in elements)

        if allVisible:
            for node, _, _, checkbox in elements:
                if isVisible(node):
                    checkbox.setChecked(False)
                    checkbox.setEnabled(False)

            for node, toggleFunc, _, _ in elements:
                if isVisible(node):
                    try:
                        toggleFunc()
                    except Exception as e:
                        print(f"[Error] Failed to hide node: {e}")
        else:
            for node, toggleFunc, _, _ in elements:
                if not isVisible(node):
                    try:
                        toggleFunc()
                    except Exception as e:
                        print(f"[Error] Failed to show node: {e}")

            for node, _, checkboxEnabler, checkbox in elements:
                isEnabled = checkboxEnabler()
                checkbox.setEnabled(isEnabled)
                checkbox.setChecked(isEnabled)

    showAllButton = qt.QPushButton("Show All")
    showAllButton.setProperty("group", "anatomy")
    showAllButton.clicked.connect(onShowAll)
    showAllButton.setMinimumHeight(24)
    showAllButton.setMaximumWidth(80)
    showAllButton.setMinimumWidth(80)
    showAllButton.setToolTip("Show or hide all anatomical elements\n• Toggles visibility of all segments\n• Enables/disables checkboxes\n• Quick way to manage all elements")
    
    toggleManipButton = qt.QPushButton("Toggle Manip")
    toggleManipButton.setProperty("group", "anatomy")
    toggleManipButton.clicked.connect(lambda: toggleManipulatorsForVisibleElements(None, None))
    toggleManipButton.setMinimumHeight(24)
    toggleManipButton.setMaximumWidth(80)
    toggleManipButton.setMinimumWidth(80)
    toggleManipButton.setToolTip("Toggle manipulators for visible elements\n• Shows/hides transform handles\n• Only affects visible segments\n• Alternative: Double-click on 2D views")
    
    controlContainer = createCompactContainer([
        showAllButton,
        toggleManipButton
    ])
    
    contentLayout.addWidget(controlContainer)

    # Anatomy buttons with checkboxes and tooltips
    def addButtonPairWithCheckboxes(layout, leftLabel, rightLabel, leftCallback, rightCallback, 
                                   leftCheckboxCallback, rightCheckboxCallback, 
                                   leftCheckboxEnabler, rightCheckboxEnabler):
        global L_Tr_Checkbox, R_Tr_Checkbox, L_Cup_Checkbox, R_Cup_Checkbox
        
        container = qt.QWidget()
        containerLayout = qt.QHBoxLayout(container)
        containerLayout.setContentsMargins(4, 4, 4, 4)
        containerLayout.setSpacing(4)
        
        # Left button and checkbox
        leftButton = qt.QPushButton(leftLabel)
        leftButton.setProperty("group", "anatomy")
        leftButton.setMaximumWidth(65)
        
        leftCheckbox = qt.QCheckBox("🔧")
        leftCheckbox.setMaximumWidth(30)
        
        # Right button and checkbox  
        rightButton = qt.QPushButton(rightLabel)
        rightButton.setProperty("group", "anatomy")
        rightButton.setMaximumWidth(65)
        
        rightCheckbox = qt.QCheckBox("🔧")
        rightCheckbox.setMaximumWidth(30)
        
        # Add tooltips based on button type
        if "Tr" in leftLabel:
            leftButton.setToolTip("Show/hide Left Trochanter\n• Creates 10mm diameter sphere\n• Positioned left side of image\n• Use checkbox to enable manipulators")
            rightButton.setToolTip("Show/hide Right Trochanter\n• Creates 10mm diameter sphere\n• Positioned right side of image\n• Use checkbox to enable manipulators")
            leftCheckbox.setToolTip("Enable manipulators for Left Trochanter\n• Shows transform handles\n• Allows manual positioning\n• Only available when segment is visible")
            rightCheckbox.setToolTip("Enable manipulators for Right Trochanter\n• Shows transform handles\n• Allows manual positioning\n• Only available when segment is visible")
        elif "Cup" in leftLabel:
            leftButton.setToolTip("Show/hide Left Cup\n• Creates hemisphere shape\n• Size controlled by spinbox below\n• Positioned left side of image")
            rightButton.setToolTip("Show/hide Right Cup\n• Creates hemisphere shape\n• Size controlled by spinbox below\n• Positioned right side of image")
            leftCheckbox.setToolTip("Enable manipulators for Left Cup\n• Shows transform handles\n• Allows manual positioning\n• Only available when segment is visible")
            rightCheckbox.setToolTip("Enable manipulators for Right Cup\n• Shows transform handles\n• Allows manual positioning\n• Only available when segment is visible")
        
        # Assign to global variables
        if "Tr" in leftLabel:
            L_Tr_Checkbox = leftCheckbox
            R_Tr_Checkbox = rightCheckbox
        elif "Cup" in leftLabel:
            L_Cup_Checkbox = leftCheckbox
            R_Cup_Checkbox = rightCheckbox
        
        # Update checkbox states
        def updateLeftCheckboxState():
            enabled = leftCheckboxEnabler()
            leftCheckbox.setEnabled(enabled)
            leftCheckbox.setChecked(enabled)
        
        def updateRightCheckboxState():
            enabled = rightCheckboxEnabler()
            rightCheckbox.setEnabled(enabled)
            rightCheckbox.setChecked(enabled)
        
        # Button callbacks
        def onLeftButtonClicked():
            leftCallback()
            updateLeftCheckboxState()
        
        def onRightButtonClicked():
            rightCallback()
            updateRightCheckboxState()
        
        # Connect signals
        leftButton.clicked.connect(onLeftButtonClicked)
        rightButton.clicked.connect(onRightButtonClicked)
        leftCheckbox.toggled.connect(lambda state: leftCheckboxCallback(state) if leftCheckbox.isEnabled() else None)
        rightCheckbox.toggled.connect(lambda state: rightCheckboxCallback(state) if rightCheckbox.isEnabled() else None)
        
        # Initialize checkbox states
        updateLeftCheckboxState()
        updateRightCheckboxState()
        
        # Add widgets to layout
        containerLayout.addWidget(leftButton)
        containerLayout.addWidget(leftCheckbox)
        containerLayout.addWidget(rightButton)
        containerLayout.addWidget(rightCheckbox)
        
        layout.addWidget(container)
    
    def addButtonWithCheckbox(layout, label, buttonCallback, checkboxLabel, checkboxCallback, checkboxEnabler):
        global Ischial_Checkbox
    
        container = qt.QWidget()
        containerLayout = qt.QHBoxLayout(container)
        containerLayout.setContentsMargins(4, 4, 4, 4)
        containerLayout.setSpacing(4)
    
        shortLabel = label.replace("Show/hide ", "").replace("Trochanter", "Tr").replace("Show ", "")
        button = qt.QPushButton(shortLabel)
        button.setProperty("group", "anatomy")
    
        checkbox = qt.QCheckBox("🔧")       
        checkbox.setMaximumWidth(60)
        
        # Add tooltip for Ischial Line
        if "Ischial" in label:
            button.setToolTip("Show/hide Ischial Reference Line\n• Creates horizontal reference line\n• Positioned 50mm below volume center\n• Used for head calculations")
            checkbox.setToolTip("Enable line handles\n• Allows manual repositioning\n• Drag endpoints to adjust\n• Important for accurate head placement")
    
        if "Ischial" in checkboxLabel:
            Ischial_Checkbox = checkbox
    
        def updateCheckboxState():
            enabled = checkboxEnabler()
            checkbox.setEnabled(enabled)
            checkbox.setChecked(enabled)
    
        def onButtonClicked():
            buttonCallback()
            qt.QTimer.singleShot(100, updateCheckboxState)
    
        button.clicked.connect(onButtonClicked)
        checkbox.toggled.connect(lambda state: checkboxCallback(state) if checkbox.isEnabled() else None)
    
        updateCheckboxState()
    
        containerLayout.addWidget(button)
        containerLayout.addWidget(checkbox)
        layout.addWidget(container)
        return onButtonClicked
    
    def canEnableLTrCheckbox():
        return L_Tr_Node is not None and L_Tr_Node.GetDisplayNode() is not None and L_Tr_Node.GetDisplayNode().GetVisibility()
    
    def canEnableRTrCheckbox():
        return R_Tr_Node is not None and R_Tr_Node.GetDisplayNode() is not None and R_Tr_Node.GetDisplayNode().GetVisibility()
    
    def canEnableLCupCheckbox():
        return L_Cup_Node is not None and L_Cup_Node.GetDisplayNode() is not None and L_Cup_Node.GetDisplayNode().GetVisibility()
    
    def canEnableRCupCheckbox():
        return R_Cup_Node is not None and R_Cup_Node.GetDisplayNode() is not None and R_Cup_Node.GetDisplayNode().GetVisibility()
    
    def canEnableIschialCheckbox():
        global Ischial_Line_Node
        if not Ischial_Line_Node:
            Ischial_Line_Node = slicer.mrmlScene.GetFirstNodeByName("Ischial Line")
        if not Ischial_Line_Node:
            return False
        displayNode = Ischial_Line_Node.GetDisplayNode()
        if not displayNode:
            return False
        return displayNode.GetVisibility()
    
    def toggleIschialHandles(state):
        global Ischial_Line_Node
        if not Ischial_Line_Node:
            Ischial_Line_Node = slicer.mrmlScene.GetFirstNodeByName("Ischial Line")
        if not Ischial_Line_Node or not Ischial_Line_Node.GetDisplayNode():
            slicer.util.errorDisplay("Ischial Line is not available. Please add it first.")
            return
        displayNode = Ischial_Line_Node.GetDisplayNode()
        displayNode.SetHandlesInteractive(state)
        print(f"[Debug] Ischial Line handles {'enabled' if state else 'disabled'}.")
    
    # Add Tr buttons (L Tr and R Tr in one row)
    addButtonPairWithCheckboxes(
        contentLayout,
        "L Tr", "R Tr",
        lambda: toggleTrVisibilityOrAdd(10, "L"),
        lambda: toggleTrVisibilityOrAdd(10, "R"),
        lambda state: enableItra(L_Tr_Transform, L_Tr_Node, "Left Tr", state),
        lambda state: enableItra(R_Tr_Transform, R_Tr_Node, "Right Tr", state),
        canEnableLTrCheckbox,
        canEnableRTrCheckbox
    )
    
    # Add Cup buttons (L Cup and R Cup in one row)
    addButtonPairWithCheckboxes(
        contentLayout,
        "L Cup", "R Cup", 
        lambda: toggleCupVisibilityOrAdd(50, "L"),
        lambda: toggleCupVisibilityOrAdd(50, "R"),
        lambda state: enableItra(L_Cup_Transform, L_Cup_Node, "Left Cup", state),
        lambda state: enableItra(R_Cup_Transform, R_Cup_Node, "Right Cup", state),
        canEnableLCupCheckbox,
        canEnableRCupCheckbox
    )
    
    # Keep Ischial Line as separate button
    addButtonWithCheckbox(contentLayout, "Show Ischial Line", lambda: addOrToggleLine("Ischial Line"), "Enable Ischial Handles", lambda state: toggleIschialHandles(state), canEnableIschialCheckbox)

    # Cup size spinboxes with tooltips
    cupSizeContainer = qt.QWidget()
    cupSizeLayout = qt.QHBoxLayout(cupSizeContainer)
    cupSizeLayout.setContentsMargins(4, 4, 4, 4)
    cupSizeLayout.setSpacing(8)

    # Left Cup Size
    lCupLabel = qt.QLabel("L Cup:")
    cupSizeLayout.addWidget(lCupLabel)

    cupSizeSpinBoxL = qt.QSpinBox()
    cupSizeSpinBoxL.setRange(36, 72)
    cupSizeSpinBoxL.setSingleStep(2)
    cupSizeSpinBoxL.setValue(50)
    cupSizeSpinBoxL.setEnabled(False)
    cupSizeSpinBoxL.valueChanged.connect(lambda val: addOrUpdateCup(val, "L"))
    cupSizeSpinBoxL.setToolTip("Left Cup Diameter (36-72mm)\n• Adjusts cup size in real-time\n• 2mm increments\n• Only enabled when cup is visible")
    cupSizeLayout.addWidget(cupSizeSpinBoxL)

    # Right Cup Size
    rCupLabel = qt.QLabel("R Cup:")
    cupSizeLayout.addWidget(rCupLabel)

    cupSizeSpinBoxR = qt.QSpinBox()
    cupSizeSpinBoxR.setRange(36, 72)
    cupSizeSpinBoxR.setSingleStep(2)
    cupSizeSpinBoxR.setValue(50)
    cupSizeSpinBoxR.setEnabled(False)
    cupSizeSpinBoxR.valueChanged.connect(lambda val: addOrUpdateCup(val, "R"))
    cupSizeSpinBoxR.setToolTip("Right Cup Diameter (36-72mm)\n• Adjusts cup size in real-time\n• 2mm increments\n• Only enabled when cup is visible")
    cupSizeLayout.addWidget(cupSizeSpinBoxR)

    contentLayout.addWidget(cupSizeContainer)

    addVerticalSpacer(contentLayout)

    # Reference side selection with tooltips
    referenceSideGroup = qt.QButtonGroup()
    referenceSideSelectionWidget = qt.QWidget()
    referenceSideLayout = qt.QHBoxLayout(referenceSideSelectionWidget)
    referenceSideLayout.setContentsMargins(0, 0, 0, 0)

    referenceSideLabel = qt.QLabel("Reference Side:")
    leftReferenceButton = qt.QRadioButton("Left")
    rightReferenceButton = qt.QRadioButton("Right")
    rightReferenceButton.setChecked(True)

    leftReferenceButton.setToolTip("Use Left side as reference\n• Head calculations based on left side\n• Reference lines show left measurements")
    rightReferenceButton.setToolTip("Use Right side as reference\n• Head calculations based on right side\n• Reference lines show right measurements")

    referenceSideGroup.addButton(leftReferenceButton)
    referenceSideGroup.addButton(rightReferenceButton)

    referenceSideLayout.addWidget(referenceSideLabel)
    referenceSideLayout.addWidget(leftReferenceButton)
    referenceSideLayout.addWidget(rightReferenceButton)

    contentLayout.addWidget(referenceSideSelectionWidget)

    # Head controls with tooltips
    def addHeadControl():
        container = qt.QWidget()
        layout = qt.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        def getSelectedSide():
            return "L" if leftReferenceButton.isChecked() else "R"

        showHideButton = qt.QPushButton("Show/Hide Head")
        showHideButton.setProperty("group", "stl")
        showHideButton.clicked.connect(lambda: toggleHeadVisibilityOrAdd(getSelectedSide()))
        showHideButton.setToolTip("Show/hide Femoral Head for selected side\n• Creates 28mm diameter sphere\n• Position calculated from cup and trochanter\n• Uses Ischial line as reference")
        layout.addWidget(showHideButton)

        updateButton = qt.QPushButton("Update Head")
        updateButton.setProperty("group", "stl")
        updateButton.clicked.connect(lambda: updateHead(getSelectedSide()))
        updateButton.setToolTip("Update Head position\n• Recalculates based on current positions\n• Applies rotational correction if enabled\n• Updates reference lines")
        layout.addWidget(updateButton)

        contentLayout.addWidget(container)

    addHeadControl()

    # Reference line button with tooltip
    referenceLineButton = qt.QPushButton("Toggle Upper Reference Line")
    referenceLineButton.setProperty("group", "stl")
    referenceLineButton.clicked.connect(lambda: toggleReferenceLine(
        "Left" if leftReferenceButton.isChecked() else "Right", 
        not isUpperReferenceLineVisible()
    ))
    referenceLineButton.setToolTip("Toggle Upper Reference Lines\n• Shows horizontal and vertical lines\n• Displays Head-Trochanter distance\n• Updates automatically with head movement")
    contentLayout.addWidget(referenceLineButton)

    addVerticalSpacer(contentLayout)

    # Advanced expandable section with tooltip
    advancedToggleButton = qt.QPushButton("▶ Advanced")
    advancedToggleButton.setProperty("group", "control")
    advancedToggleButton.setStyleSheet("""
        QPushButton[group="control"] {
            text-align: left;
            padding-left: 8px;
            font-weight: bold;
        }
    """)
    advancedToggleButton.setToolTip("Show/hide advanced functions\n• Position correction tools\n• Reference line adjustments\n• STL import and scaling")
    
    # Container for advanced functions (initially hidden)
    advancedFunctionsContainer = qt.QWidget()
    advancedFunctionsLayout = qt.QVBoxLayout(advancedFunctionsContainer)
    advancedFunctionsLayout.setContentsMargins(10, 5, 0, 5)
    advancedFunctionsLayout.setSpacing(3)
    advancedFunctionsContainer.setVisible(False)
    
    # Track expanded state
    isAdvancedExpanded = [False]
    
    def toggleAdvancedSection():
        isAdvancedExpanded[0] = not isAdvancedExpanded[0]
        
        if isAdvancedExpanded[0]:
            advancedToggleButton.setText("▼ Advanced")
            advancedFunctionsContainer.setVisible(True)
            print("[Debug] Advanced section expanded")
        else:
            advancedToggleButton.setText("▶ Advanced")
            advancedFunctionsContainer.setVisible(False)
            print("[Debug] Advanced section collapsed")
        
        contentWidget.adjustSize()
        slicer.app.processEvents()
    
    advancedToggleButton.clicked.connect(toggleAdvancedSection)
    contentLayout.addWidget(advancedToggleButton)
    
    # Position correction controls with tooltips
    def addPositionCorrectionControl():
        container = qt.QWidget()
        layout = qt.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        correctionToggle = qt.QRadioButton("Rotational Correction")
        correctionToggle.setToolTip("Enable Rotational Correction\n• Corrects for patient rotation\n• Calculates angle difference from Ischial line\n• Improves head positioning accuracy")
        correctionToggle.toggled.connect(lambda state: toggleRotationalCorrection(state))
        layout.addWidget(correctionToggle)

        calculateButton = qt.QPushButton("Show/hide corrected Tr")
        calculateButton.setProperty("group", "action")
        calculateButton.setToolTip("Show/hide corrected Trochanter positions\n• Displays calculated corrections\n• Shows vertical shift values\n• Based on selected reference side")
        calculateButton.clicked.connect(calculate_and_display_corrected_positions)
        layout.addWidget(calculateButton)

        advancedFunctionsLayout.addWidget(container)

    def toggleRotationalCorrection(state):
        global rotationalCorrectionEnabled
        rotationalCorrectionEnabled = state
        print(f"[Debug] Rotational Correction Enabled: {rotationalCorrectionEnabled}")

    addPositionCorrectionControl()

    # Offset controls with tooltips
    def updateReferenceLineOffset(value):
        global referenceLineOffset
        referenceLineOffset = value
        print(f"[Debug] Reference line offset updated to {referenceLineOffset} mm.")

    def updateHorizontalLineLength(value):
        global horizontalLineLength
        horizontalLineLength = value
        print(f"[Debug] Horizontal line length updated to {horizontalLineLength} mm.")

    # Vertical Offset controls
    verticalSpinBox = qt.QSpinBox()
    verticalSpinBox.setRange(0, 100)
    verticalSpinBox.setSingleStep(1)
    verticalSpinBox.setValue(referenceLineOffset)
    verticalSpinBox.valueChanged.connect(updateReferenceLineOffset)
    verticalSpinBox.setToolTip("Vertical Reference Line Offset (0-100mm)\n• Distance below head center\n• Affects reference line position\n• Default: 35mm")

    verticalOffsetContainer = createCompactContainer([
        qt.QLabel("Vertical Offset (mm): "),
        verticalSpinBox
    ])
    advancedFunctionsLayout.addWidget(verticalOffsetContainer)

    # Horizontal Offset controls
    horizontalSpinBox = qt.QSpinBox()
    horizontalSpinBox.setRange(0, 100)
    horizontalSpinBox.setSingleStep(1)
    horizontalSpinBox.setValue(horizontalLineLength)
    horizontalSpinBox.valueChanged.connect(updateHorizontalLineLength)
    horizontalSpinBox.setToolTip("Horizontal Reference Line Length (0-100mm)\n• Length of horizontal reference\n• Default: 35mm")

    horizontalOffsetContainer = createCompactContainer([
        qt.QLabel("Horizontal Offset (mm): "),
        horizontalSpinBox
    ])
    advancedFunctionsLayout.addWidget(horizontalOffsetContainer)

    # Lower reference line button with tooltip
    lowerReferenceLineButton = qt.QPushButton("Toggle Lower Reference Line")
    lowerReferenceLineButton.setProperty("group", "control")
    lowerReferenceLineButton.clicked.connect(lambda: toggleLowerReferenceLine(
        "Left" if leftReferenceButton.isChecked() else "Right"))
    lowerReferenceLineButton.setToolTip("Toggle Lower Reference Line\n• 45-degree angled line\n• Shows reamer orientation\n• Based on selected side")
    advancedFunctionsLayout.addWidget(lowerReferenceLineButton)

    # STL side selection with tooltips
    sideGroup = qt.QButtonGroup()
    sideSelectionWidget = qt.QWidget()
    sideLayout = qt.QHBoxLayout(sideSelectionWidget)
    sideLayout.setContentsMargins(0, 0, 0, 0)

    sideLayout.addWidget(qt.QLabel("Side:"))
    rightButton = qt.QRadioButton("Right")
    leftButton = qt.QRadioButton("Left")
    rightButton.setChecked(True)
    
    rightButton.setToolTip("Import STL on Right side\n• Mirrors geometry for right side\n• Positioned 5cm right of center")
    leftButton.setToolTip("Import STL on Left side\n• Original geometry orientation\n• Positioned 5cm left of center")
    
    sideGroup.addButton(rightButton)
    sideGroup.addButton(leftButton)
    sideLayout.addWidget(rightButton)
    sideLayout.addWidget(leftButton)

    advancedFunctionsLayout.addWidget(sideSelectionWidget)

    # Function to update side dynamically
    def updateSide():
        selectedSide = "Right" if rightButton.isChecked() else "Left"
        importSTLAsSegmentation(side=selectedSide, updateExisting=True)

    rightButton.toggled.connect(updateSide)
    leftButton.toggled.connect(updateSide)

    # STL import controls with tooltips
    stlImportContainer = qt.QWidget()
    stlImportLayout = qt.QHBoxLayout(stlImportContainer)
    stlImportLayout.setContentsMargins(0, 0, 0, 0)

    importSTLButton = qt.QPushButton("Import STL")
    importSTLButton.setProperty("group", "stl")
    importSTLButton.clicked.connect(
        lambda: importSTLAsSegmentation(side="Right" if rightButton.isChecked() else "Left")
    )
    importSTLButton.setToolTip("Import STL file for implant\n• Supports CSV size mapping\n• Auto-enables manipulators\n• Positioned relative to volume center")
    stlImportLayout.addWidget(importSTLButton)

    manipulatorCheckbox = qt.QCheckBox("Enable Manipulator")
    manipulatorCheckbox.setChecked(True)
    manipulatorCheckbox.setEnabled(False)
    manipulatorCheckbox.setToolTip("Enable STL manipulators\n• Shows transform handles for imported STL\n• Allows manual positioning\n• Enabled after successful import")
    stlImportLayout.addWidget(manipulatorCheckbox)

    advancedFunctionsLayout.addWidget(stlImportContainer)

    # Wrapper to enable checkbox after successful import
    def enableCheckboxAfterImport():
        global currentSegmentationNode
        if currentSegmentationNode:
            manipulatorCheckbox.setEnabled(True)
            print("[Debug] Checkbox enabled after successful STL import.")
        else:
            print("[Warning] STL import failed or segmentation node not created.")

    originalImportFunction = importSTLButton.clicked
    importSTLButton.clicked.connect(lambda: enableCheckboxAfterImport())

    # Connect the checkbox state to toggle manipulator visibility
    def onCheckboxToggled(checked):
        if currentSegmentationNode:
            transformNodeID = currentSegmentationNode.GetTransformNodeID()
            if transformNodeID:
                transformNode = slicer.mrmlScene.GetNodeByID(transformNodeID)
                if transformNode:
                    displayNode = transformNode.GetDisplayNode()
                    if displayNode:
                        displayNode.SetEditorVisibility(checked)
                        print(f"Manipulator visibility set to: {checked}")

    manipulatorCheckbox.toggled.connect(onCheckboxToggled)

    # STL scaling controls with tooltips
    stlScaleContainer = createCompactContainer([
        qt.QPushButton("Scale stem UP"),
        qt.QPushButton("Scale stem DOWN")
    ])
    
    stlScaleButtons = stlScaleContainer.findChildren(qt.QPushButton)
    for button in stlScaleButtons:
        button.setProperty("group", "control")
    
    stlScaleButtons[0].setToolTip("Increase STL size\n• Steps through predefined sizes\n• Proportional scaling\n• Updates size label")
    stlScaleButtons[1].setToolTip("Decrease STL size\n• Steps through predefined sizes\n• Proportional scaling\n• Updates size label")
    
    stlScaleButtons[0].clicked.connect(lambda: scaleCurrentSegmentationBoundingBox(increment=True))
    stlScaleButtons[1].clicked.connect(lambda: scaleCurrentSegmentationBoundingBox(increment=False))
    
    advancedFunctionsLayout.addWidget(stlScaleContainer)

    # Add the advanced functions container to the main layout
    contentLayout.addWidget(advancedFunctionsContainer)

    addVerticalSpacer(contentLayout)

    # Final action buttons with tooltips
    actionContainer = createCompactContainer([
        qt.QPushButton("X-ray Mode"),
        qt.QPushButton("Reset Scene"),
        qt.QPushButton("Close")
    ])
    
    actionButtons = actionContainer.findChildren(qt.QPushButton)
    for button in actionButtons:
        button.setProperty("group", "action")
    
    actionButtons[0].setToolTip("Toggle X-ray Mode\n• First press: Apply X-ray transform and show Coronal view\n• Second press: Undo transform and restore Four-Up view\n• Automatically adjusts scaling")
    actionButtons[1].setToolTip("Reset Scene\n• Removes all segments and transforms\n• Preserves X-ray transform if applied\n• Resets spinboxes to default values")
    actionButtons[2].setToolTip("Close Toolbar\n• Removes the Easy Hip Planner toolbar\n• Does not affect scene content\n• Can be reopened by running script again")
    
    actionButtons[0].clicked.connect(applyXRayTransform)
    actionButtons[1].clicked.connect(resetScene)
    actionButtons[2].clicked.connect(lambda: slicer.util.mainWindow().removeToolBar(toolbar))
    
    contentLayout.addWidget(actionContainer)

    # Set the content widget to the scroll area
    scrollArea.setWidget(contentWidget)
    
    # Add the scroll area to the toolbar
    toolbar.addWidget(scrollArea)

    # Apply styles
    applyModernToolbarStyles(toolbar)
    
    print("[Debug] Modern scrollable toolbar created successfully with tooltips.")

# Wywołanie funkcji do stworzenia paska
createCupToolbar()

def validateDisplayNodes():
    """
    Validate and ensure display nodes are present for all segmentation nodes.
    """
    for segmentNode in slicer.util.getNodesByClass("vtkMRMLSegmentationNode"):
        if not segmentNode.GetDisplayNode():
            print(f"[Warning] Missing Display Node for segmentation: {segmentNode.GetName()}")
            # Create and assign a new Display Node
            displayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationDisplayNode", f"{segmentNode.GetName()} Display")
            segmentNode.SetAndObserveDisplayNodeID(displayNode.GetID())
            displayNode.SetVisibility3D(True)
            displayNode.SetVisibility2DFill(True)
            displayNode.SetVisibility2DOutline(True)
            print(f"[Debug] Created and assigned new Display Node for: {segmentNode.GetName()}")
        else:
            print(f"[Debug] Display Node exists for segmentation: {segmentNode.GetName()}")
