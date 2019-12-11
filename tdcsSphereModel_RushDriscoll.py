#!/usr/bin/python3

import sys
import os
import numpy as np
import nibabel as nib

from scipy import special as scsp

from PySide2.QtCore import  \
    Signal,                 \
    Slot

from PySide2.QtGui import   \
    QDoubleValidator,       \
    QImage,                 \
    QColor,                 \
    qRgb,                   \
    QPixmap

from PySide2.QtWidgets import   \
    QApplication,               \
    QMainWindow,                \
    QWidget,                    \
    QMenu,                      \
    QMenuBar,                   \
    QFileDialog,                \
    QVBoxLayout,                \
    QHBoxLayout,                \
    QDialog,                    \
    QFormLayout,                \
    QLineEdit,                  \
    QPushButton,                \
    QDoubleSpinBox,             \
    QLabel,                     \
    QSlider,                    \
    QComboBox,                  \
    QInputDialog,               \
    QLineEdit

'''
    class : AppView
        Creates the main window of the application
'''
class AppView( QWidget, object ):

    mApp       = None
    mMenu      = None
    mImagePane = None

    mOutFilePath = None
    

    #mConductivities = [1., 1., 1.]
    #mRadii          = [8., 1., 2.]

    DEFAULT_WINDOW_EXTENDS = 256

    def __init__( self ):
        super( AppView, self).__init__();
        
        self.mApp = MainApp( self.DEFAULT_WINDOW_EXTENDS )
        
        # create GUI components
        self.mMenu = MyMenuBar()
        def exportEPot( path ):
            self.mExportAsNii( self.mApp.getEPot(), path + "/sphere_model_epot.nii" )
        def exportEF( path ):
            self.mExportAsNii( self.mApp.getEF(), path + "/sphere_model_ef.nii" )
        def updateRadii( skin_r, skull_r, csf_r ):
            self.mApp.setRadii( skin_r, skull_r, csf_r )
            self.mApp.initCoordinateSystem( self.DEFAULT_WINDOW_EXTENDS )
        self.mMenu.exportepot.connect( exportEPot )
        self.mMenu.exportef.connect( exportEF )
        self.mMenu.closeappselected.connect( self.close )
        self.mMenu.conductivitiesselected.connect( self.mApp.setConductivities )
        self.mMenu.radiiselected.connect( updateRadii )
       
        self.mImagePane = ImagePane()
        self.mImagePane.updateImage( self.mApp.getData()[int(self.DEFAULT_WINDOW_EXTENDS / 2)] )
        
        self.mSliceSlider = QSlider()
        self.mSliceSlider.setMinimum(0)
        self.mSliceSlider.setMaximum( self.DEFAULT_WINDOW_EXTENDS - 1)
        self.mSliceSlider.setPageStep(1)
        self.mSliceSlider.setValue( int( self.DEFAULT_WINDOW_EXTENDS / 2))
        
        self.mBrightnessSlider = QSlider()
        self.mBrightnessSlider.setMinimum(0)
        self.mBrightnessSlider.setMaximum(255)
        self.mBrightnessSlider.setPageStep(1)
        self.mBrightnessSlider.setValue( 0 )
        
        def sliderHandler():
            self.mImagePane.updateImage( self.mApp.getData()[self.mSliceSlider.value()], self.mBrightnessSlider.value() )
        def brightnessSliderHandler():
            self.mImagePane.updateImage( self.mApp.getData()[self.mSliceSlider.value()], self.mBrightnessSlider.value() )
        self.mSliceSlider.valueChanged.connect( sliderHandler )
        self.mBrightnessSlider.valueChanged.connect( brightnessSliderHandler )
        
        def switchFieldsToDisplay( index ):
            if index == 0:
                self.mApp.activateEPot()
            elif index == 1:
                self.mApp.activateEF()
            self.mImagePane.updateImage( self.mApp.getData()[self.mSliceSlider.value()], self.mBrightnessSlider.value())
        self.mToolbar = MyToolbar()
        self.mFieldChooser = QComboBox()
        self.mFieldChooser.addItem(Labels.LBL_EPot)
        self.mFieldChooser.addItem(Labels.LBL_EF)
        self.mFieldChooser.currentIndexChanged.connect( switchFieldsToDisplay )
        self.mFieldChooser.setEnabled( False )

        def calcEPotButtonHandler(a,b,c,d):
            self.mApp.calcEPot(a,b,c,d)
            self.mImagePane.updateImage( self.mApp.getData()[self.mSliceSlider.value()])
        self.mToolbar.executeepotcalcappsignal.connect( calcEPotButtonHandler )
        def calcEFButtonHandler():
            self.mApp.calcEF()
            self.mFieldChooser.setEnabled( True )
            self.mFieldChooser.setCurrentIndex( 1 )
            self.mImagePane.updateImage( self.mApp.getData()[self.mSliceSlider.value()])
            
        self.mToolbar.executeefcalcappsignal.connect( calcEFButtonHandler )

        # define layout
        outerlayout = QHBoxLayout()

        layout = QVBoxLayout()
        layout.setMenuBar( self.mMenu )
 
        layout.addWidget( self.mImagePane )
        layout.addWidget( self.mFieldChooser )
        layout.addWidget( self.mToolbar )
        
        outerlayout.addLayout( layout )
        outerlayout.addWidget( self.mBrightnessSlider )
        outerlayout.addWidget( self.mSliceSlider )

        self.setLayout( outerlayout )

        # prepare window and show
        self.resize( self.DEFAULT_WINDOW_EXTENDS, self.DEFAULT_WINDOW_EXTENDS )
        self.setWindowTitle( Labels.LBL_MA_WINDOW_TITLE )
        self.show()

    def mExportAsNii( self, data, path ):
        try:
            newImage = nib.Nifti1Image( data, np.diag([1,1,1,1]) )
            newImage.header.set_intent( 1001 ) # 1001 = estimate
            newImage.header.set_data_shape( [self.DEFAULT_WINDOW_EXTENDS]*3 )
            newImage.update_header()
            nib.save( newImage, path )
        except IOError:
            print( "Error while creating output file." )

class MyToolbar(QWidget):

    executeepotcalcappsignal = Signal(float,float,float,float)
    executeefcalcappsignal = Signal()

    mElectrodePosInputs = None

    btEpot = None
    btEF   = None

    def __init__(self):
        QWidget.__init__(self)
        self.mElectrodePosInputs = [None] * 2
        for electrodeNum in range(0,2): # loop over both electrodes
            self.mElectrodePosInputs[ electrodeNum ] = [None] * 2
            for dimension in range(0,2):        # loop over both angles defining the position
                self.mElectrodePosInputs[electrodeNum][dimension] = QDoubleSpinBox()
                # theta € [0,np.pi]
                # phi   € [0,2*np.pi]
                if dimension == 0:
                    self.mElectrodePosInputs[electrodeNum][dimension].setMaximum( 180 )
                else:
                    self.mElectrodePosInputs[electrodeNum][dimension].setMaximum( 360 )
                self.mElectrodePosInputs[electrodeNum][dimension].setMinimum( 0 )
                self.mElectrodePosInputs[electrodeNum][dimension].setSingleStep( 0.1 )
                self.mElectrodePosInputs[electrodeNum][dimension].setDecimals( 3 )

        # set defaults: electrodes on opposing sites at 180 degree angle
        self.mElectrodePosInputs[0][0].setValue( 90 )
        self.mElectrodePosInputs[0][1].setValue( 0 )
        self.mElectrodePosInputs[1][0].setValue( 90 )
        self.mElectrodePosInputs[1][1].setValue( 180 )


        self.btEPot = QPushButton( Labels.LBL_ELPOT_BT, self )
        self.btEPot.clicked.connect( self.emitEPotButtonSignalToMainApp )
        self.btEF = QPushButton( Labels.LBL_EFIELD_BT, self )
        self.btEF.clicked.connect( self.emitEFButtonSignalToMainApp )
        self.btEF.setEnabled( False )

        # assemble layout
        layout = QVBoxLayout()
        columns = [QHBoxLayout(), QHBoxLayout()]
        for electrodeNum in range(0,2):
            columns[electrodeNum].addWidget( QLabel( Labels.LBL_ELECTRODE + " " + str(electrodeNum+1) ) )
            columns[electrodeNum].addWidget( QLabel( Labels.LBL_THETA ) )
            columns[electrodeNum].addWidget( self.mElectrodePosInputs[electrodeNum][0] )
            columns[electrodeNum].addWidget( QLabel( Labels.LBL_PHI ) )
            columns[electrodeNum].addWidget( self.mElectrodePosInputs[electrodeNum][1] )

        columns[0].addWidget( self.btEPot )
        columns[1].addWidget( self.btEF )

        layout.addLayout(columns[0])
        layout.addLayout(columns[1])


        self.setLayout( layout )

    def emitEPotButtonSignalToMainApp(self):
        self.executeepotcalcappsignal.emit( (self.mElectrodePosInputs[0][0].value() / 180. * np.pi),
                                            (self.mElectrodePosInputs[0][1].value() / 180. * np.pi),
                                            (self.mElectrodePosInputs[1][0].value() / 180. * np.pi),
                                            (self.mElectrodePosInputs[1][1].value() / 180. * np.pi) )
        self.btEF.setEnabled( True )

    def emitEFButtonSignalToMainApp(self):
        self.executeefcalcappsignal.emit() 



class MyMenuBar(QMenuBar):
    exportepot = Signal(str)
    exportef = Signal(str)
    closeappselected = Signal()
    radiiselected = Signal(float,float,float)
    conductivitiesselected = Signal(float,float,float)

    mRadii = [ 92, 85, 80 ]
    mConductivities = [ 0.465, 0.01, 0.33 ]

    def __init__( self ):
        QMenuBar.__init__(self)

        fileMenu = QMenu( Labels.LBL_FILE_MENU, self )
        exportEPotAction = fileMenu.addAction( Labels.LBL_EXPORT_EPOT )
        exportEFAction = fileMenu.addAction( Labels.LBL_EXPORT_EF )
        exitAction = fileMenu.addAction( Labels.LBL_EXIT )
        
        exportEFAction.triggered.connect( self.chooseOutputFile )
        exportEPotAction.triggered.connect( self.chooseOutputFile )
        exitAction.triggered.connect( self.emitCloseSignalToMainApp  )
        
        settingsMenu = QMenu( Labels.LBL_SETTINGS_MENU, self )
        setRadiiAction = settingsMenu.addAction( Labels.LBL_SET_RADII )
        setConductivitiesAction = settingsMenu.addAction( Labels.LBL_SET_CONDUCTIVITIES )
        setRadiiAction.triggered.connect( self.setRadiiDialog )
        setConductivitiesAction.triggered.connect( self.setConductivitiesDialog )
   
        self.addMenu( fileMenu )
        self.addMenu( settingsMenu )

    def emitCloseSignalToMainApp(self):
        self.closeappselected.emit()

    def chooseOutputFile( self ):
        outFilePath = QFileDialog.getExistingDirectory( self, Labels.LBL_SAVE_RESULT, os.getcwd() )

        if outFilePath is not None:
            if self.sender().text() == Labels.LBL_EXPORT_EPOT:
                self.exportepot.emit( outFilePath )
            elif self.sender().text() == Labels.LBL_EXPORT_EF:
                self.exportef.emit( outFilePath )

    def setRadiiDialog( self ):
        dialog = MultiValueInputDialog( 
            [ 
                {"label":"skin", "default":str( self.mRadii[0] ) },
                {"label":"skull", "default":str( self.mRadii[1] ) },
                {"label":"brain", "default":str( self.mRadii[2] ) }
            ], True )
        def radiiFormFilledHandler( ):
            self.mRadii = dialog.getValues()
            try:
                self.radiiselected.emit( float(self.mRadii[0]), float(self.mRadii[1]), float(self.mRadii[2]) )
            except ValueError:
                print( "Invalid radius values." )
        
        dialog.accepted.connect( radiiFormFilledHandler )
        dialog.exec_()

    def setConductivitiesDialog( self ):
        dialog = MultiValueInputDialog( 
            [ 
                {"label":"skin", "default": str( self.mConductivities[0] )},
                {"label":"skull", "default":str( self.mConductivities[1] )},
                {"label":"brain", "default":str( self.mConductivities[2] )}
            ], True )
        def conductivitiesFormFilledHandler( ):
            self.mConductivities = dialog.getValues()
            try:
                self.conductivitiesselected.emit( float(self.mConductivities[0]), float(self.mConductivities[1]), float(self.mConductivities[2]) )
            except ValueError:
                print( "Invalid conductivity values." )
                
        dialog.accepted.connect( conductivitiesFormFilledHandler )
        dialog.exec_()

class ImagePane(QWidget):
    
    mImageHolder = None

    def __init__( self ):
        QWidget.__init__(self)
        # we use an empty QLabel here and assing a pixmap later
        self.mImageHolder = QLabel()

        # make the label center within the image-pane
        layout = QHBoxLayout( )
        layout.addStretch()
        layout.addWidget( self.mImageHolder )
        layout.addStretch()

        self.setLayout( layout )

    #
    # Converts the 2D data-array into an image displayed within the 'imageHolder'
    #   -> We have a QImage inside a pixmap inside a QLabel.
    # @param _imageData  : the data-array, numbers between [0,255]
    # @return the created QImage
    #
    def updateImage( self, data, addedBrightness = 0 ):
        # QImage allows pixel-wise modification of the canvas
        image  = QImage( len( data ), len( data[0] ), QImage.Format_RGB32 )
        
        data_max = np.amax( data )
        data_min = abs(np.amin( data ))
        
        dataToDisplay = None

        # scale the data to its maximum
        if data_max > data_min :
            dataToDisplay = data / data_max
            dataToDisplay *= 255
        elif data_min > 0:
            dataToDisplay = data / data_min
            dataToDisplay *= 255
        else:
            dataToDisplay = data

        scaleFactor = 1.0


        log_scale_factor = 255/np.log(255)
        for y in range( 0, len( dataToDisplay[0]) ):
            for x in range( 0, len( dataToDisplay ) ):
                '''
                non-logarithmic
                intensity = min( abs(int(dataToDisplay[x][y])) + addedBrightness, 255)
                if dataToDisplay[x][y] > 0 :
                    image.setPixel( x, y, qRgb( intensity, 0, 0) )
                else:
                    image.setPixel( x, y, qRgb( 0, 0, intensity) )
                '''
                intensity = min( abs(int(dataToDisplay[x][y])) + 1 + addedBrightness, 255)
                if dataToDisplay[x][y] > 0 :
                    image.setPixel( x, y, qRgb( int(np.log(intensity)*log_scale_factor), 0, 0) )
                else:
                    image.setPixel( x, y, qRgb( 0, 0, int(np.log(intensity)*log_scale_factor) ) )


        # QLabels may only get QPixmpas assigned, no QImages
        # hence we wrap the QImage inside the pixmap
        pixmap = QPixmap( )
        pixmap.convertFromImage( image )

        pixmap = pixmap.scaled( pixmap.width()  * scaleFactor,
                                pixmap.height() * scaleFactor )


        self.mImageHolder.setPixmap( pixmap )

        self.update();
        


class MultiValueInputDialog(QDialog):

    mInputFields = []

    #
    # entry ... list of dicts: label(String),default(any)
    #
    def __init__( self, entries, only_numeric, defaults=None ):
        QDialog.__init__(self)

        # create layout
        layout = QFormLayout()


        # add GUI elements to layout
        self.mInputFields = []
        defaultValue = "0"
        for entry in entries:
            if entry["default"] != None:
                defaultValue = entry["default"]
            self.mInputFields.append( QLineEdit( defaultValue, self ) )
            if only_numeric == True:
                self.mInputFields[-1].setValidator( QDoubleValidator(0, 100, 2, self ) )

            layout.addRow( entry["label"], self.mInputFields[-1] )

        okButton = QPushButton("OK")
        cancelButton = QPushButton("Cancel")

        okButton.clicked.connect( self.accept )
        cancelButton.clicked.connect( self.reject )

        layout.addRow( okButton, cancelButton )

        self.setLayout( layout )

    def getValues( self ):
        return [ entry.text() for entry in self.mInputFields ] 

'''
    class : Labels
        contains all string used in the GUI
'''
class Labels:

    LBL_MA_WINDOW_TITLE      = "3-layered sphere model simulation"
    LBL_FILE_MENU            = "File"
    LBL_EXIT                 = "Quit"
    LBL_SETTINGS_MENU        = "Settings"
    LBL_SET_RADII            = "Set sphere radii"
    LBL_SET_CONDUCTIVITIES   = "Set conductivities"
    LBL_ELPOT_BT             = "Calc ElPot"
    LBL_EFIELD_BT            = "Calc E-Field"
    LBL_PHI                  = "Phi"
    LBL_THETA                = "Theta"
    LBL_ELECTRODE            = "Electrode"
    LBL_EXPORT_EPOT          = "Export ElPot as NIfTI"
    LBL_EXPORT_EF            = "Export EField as NIfTI"
    LBL_EPot                 = "ElPot"
    LBL_EF                   = "EField Strength"
    LBL_SAVE_RESULT          = "Export result"

'''
    class : MainApp
        containing the application logic
'''
class MainApp:
    mSkinCond    = None
    mSkullCond   = None
    mBrainCond   = None
    mSkinRadius  = None
    mSkullRadius = None
    mBrainRadius = None
    mSkinRadius_normalized  = None
    mSkullRadius_normalized = None
    mBrainRadius_normalized = None
    mAngularDiffElectrode1_grid = None
    mAngularDiffElectrode2_grid = None
    mElectrodeCoords = None

    mExtent          = None
    mCenterOffset    = None
    mInputCurrent    = None

    mBrainSkullCondRatio = None
    mSkullSkinCondRatio = None

    mRadius_grid = None
    mTheta_grid  = None
    mPhi_grid    = None

    mActiveField = None
    mEPot        = None
    mEF          = None

    mNormalizeRadii = None

    def __init__(self, extent):
        self.mElectrodeCoords = []
        
        # Intialize parameters (some may be overriden by GUI params)
        self.writeElectrodeCoordinate( np.pi / 2., 0, 1)
        self.writeElectrodeCoordinate( np.pi / 2., np.pi, 2)
        self.mInputCurrent = 0.001      # milliampere
        self.mMaxIteration = 5          # 50 yields reasonable results
        self.mNormalizeRadii = False    # does not compute the correct results with normalized radii
        # according to the paper and https://www.researchgate.net/figure/Parameters-in-the-three-sphere-model_tbl1_21226863
        self.setRadii( skin=92, skull=85, brain=80 )
        self.setConductivities( skin=0.465, skull=0.01, brain=0.33 )
        
        # initialize members
        self.initCoordinateSystem( extent )
        self.mEPot = np.zeros([extent,extent,extent])
        self.mActiveField = self.mEPot

    def initCoordinateSystem( self, extent ): 
        axis_dimension = np.linspace(-1*self.mSkinRadius_normalized,self.mSkinRadius_normalized,extent)
        xx,yy,zz = np.meshgrid( axis_dimension, axis_dimension, axis_dimension )

        self.mRadius_grid = np.sqrt( xx**2 + yy**2 + zz**2 )
        self.mTheta_grid  = np.arccos( zz / self.mRadius_grid )
        self.mPhi_grid    = np.arctan2( yy , xx )
        
        self.mCenterOffset = int( extent / 2 )
    
    def initElectrodeAngleDiff( self ):
        # from : https://math.stackexchange.com/a/2384360
        self.mAngularDiffElectrode1_grid = \
            np.nan_to_num( np.arccos( \
                np.cos( self.mElectrodeCoords[0][0] ) * np.cos( self.mTheta_grid ) \
                + \
                np.sin( self.mElectrodeCoords[0][0] ) * np.sin( self.mTheta_grid ) * np.cos( self.mPhi_grid - self.mElectrodeCoords[0][1]) \
            ) )
        self.mAngularDiffElectrode2_grid = \
            np.nan_to_num( np.arccos( \
                np.cos( self.mElectrodeCoords[1][0] ) * np.cos( self.mTheta_grid ) \
                + \
                np.sin( self.mElectrodeCoords[1][0] ) * np.sin( self.mTheta_grid ) * np.cos( self.mPhi_grid - self.mElectrodeCoords[1][1]) \
            ) )


    # theta € [0,np.pi]
    # phi   € [0,2*np.pi]
    def electrodePosInGrid( self, theta, phi, radius=1. ):
        assert radius <= 1. and radius > 0  # radius must be within the interval (0,1]

        x = (self.mCenterOffset) * (radius * np.sin( theta ) * np.cos( phi ) + 1)
        y = (self.mCenterOffset) * (radius * np.sin( theta ) * np.sin( phi ) + 1)
        z = (self.mCenterOffset) * (radius * np.cos( theta ) + 1)

        return x,y,z

    #
    # Compute the angle between two points given in spherical coordinates
    #
    # coords_1 ... tuple, (theta, phi)
    # coords_2 ... tuple, (theta, phi)
    #
    def angularDiff( self, coords_1, coords_2 ):
        return np.arccos( np.cos( coords_1[0] ) * np.cos( coords_2[0] ) + np.sin( coords_1[0] ) * np.sin( coords_2[0] ) * np.cos( coords_1[1] - coords_2[1] ) )

    def A( self, n ):
        val = np.nan_to_num( ( \
                (( 2 * n + 1)**3) / (2*n) \
               ) \
                 / \
               ( \
                ( ( self.mBrainSkullCondRatio + 1 ) * n + 1 ) * ( ( self.mSkullSkinCondRatio + 1 ) * n + 1 ) \
                 + \
                ( ( self.mBrainSkullCondRatio - 1 ) * ( self.mSkullSkinCondRatio - 1 ) * n * (n + 1) * ((self.mBrainRadius_normalized / self.mSkullRadius_normalized )**(2*n+1)) ) \
                 + \
                ( ( self.mSkullSkinCondRatio - 1 ) * (n + 1) * ( ( self.mBrainSkullCondRatio + 1 )*n + 1 )  * ( ( self.mSkullRadius_normalized / self.mSkinRadius_normalized )**(2*n+1)) ) \
                 + \
                ( ( self.mBrainSkullCondRatio - 1 ) * (n + 1) * ( ( self.mSkullSkinCondRatio + 1 ) * (n + 1) - 1 ) * (( self.mBrainRadius_normalized / self.mSkinRadius_normalized )**(2*n+1)) ) \
               ) )

        return val

    def S( self, n ):
        val =  np.nan_to_num( \
                ( self.A(n) / ( (self.mSkinRadius_normalized**n) * ( 2*n + 1 ) ) ) * ( ( 1 + self.mBrainSkullCondRatio ) * n + 1 ) \
               )

        return val

    def U( self, n ):
        val =  np.nan_to_num( \
                 ( self.A(n) / ( (self.mSkinRadius_normalized**n) * ( 2*n + 1 ) ) ) * n * ( 1 - self.mBrainSkullCondRatio ) * ( self.mBrainRadius_normalized**(2*n+1) )  \
                )

        return val

    def T( self, n ):
        val = np.nan_to_num( ( self.A(n) / ( (self.mSkinRadius_normalized**n) * ( ( 2*n + 1 )**2) ) ) \
                * \
               ( \
                ( ( 1 + self.mBrainSkullCondRatio ) * n + 1 ) \
                  * \
                ( ( 1 + self.mSkullSkinCondRatio ) * n + 1 ) \
                  + \
                 ( n * ( n + 1 ) * ( 1 - self.mBrainSkullCondRatio ) * ( 1 - self.mSkullSkinCondRatio ) * ( ( self.mBrainRadius_normalized / self.mSkullRadius_normalized )**(2*n+1) ) ) \
               ) )

        return val
    
    def W( self, n ):
        val = np.nan_to_num( ( ( n * self.A(n) ) / ( (self.mSkinRadius**n) * ( ( 2*n + 1 )**2 ) ) ) \
                * \
               ( \
                 ( 1 - self.mSkullSkinCondRatio ) * ( ( 1 + self.mBrainSkullCondRatio ) * n + 1 ) * ( self.mSkullRadius_normalized**(2*n+1) ) \
                 + \
                 ( 1 - self.mBrainSkullCondRatio ) * ( ( 1 + self.mSkullSkinCondRatio ) * n + self.mSkullSkinCondRatio ) * ( self.mBrainRadius_normalized**(2*n+1) ) \
               ) )

        return val
    
    @Slot(float, float, float, float)
    def calcEPot( self, electrodeAtheta, electrodeAphi, electrodeBtheta, electrodeBphi ):
        self.writeElectrodeCoordinate( electrodeAtheta, electrodeAphi, 0) 
        self.writeElectrodeCoordinate( electrodeBtheta, electrodeBphi, 1)

        self.initElectrodeAngleDiff()

        self.mBrainSkullCondRatio = self.mBrainCond / self.mSkullCond
        self.mSkullSkinCondRatio  = self.mSkullCond / self.mSkinCond

        summation_factor = self.mInputCurrent / ( 2 * np.pi * self.mSkinCond * self.mSkinRadius_normalized)

        buffer_region = (np.asarray( self.mRadius_grid <= 1. ) & np.asarray( self.mRadius_grid > self.mSkinRadius_normalized)).nonzero()
        skin_region   = (np.asarray( self.mRadius_grid <= self.mSkinRadius_normalized ) & np.asarray( self.mRadius_grid > self.mSkullRadius_normalized)).nonzero()
        skull_region  = (np.asarray( self.mRadius_grid <= self.mSkullRadius_normalized ) & np.asarray( self.mRadius_grid > self.mBrainRadius_normalized )).nonzero()
        brain_region  = np.asarray( self.mRadius_grid <= self.mBrainRadius_normalized ).nonzero()
    
        for n in range( 1, self.mMaxIteration ):
            electrode_diff_legendre = \
                np.nan_to_num( \
                scsp.legendre(n)( np.cos( self.mAngularDiffElectrode1_grid ) ) \
                - \
                scsp.legendre(n)( np.cos( self.mAngularDiffElectrode2_grid ) ) \
                )
            self.mEPot[ brain_region ] += \
                self.A(n) \
                * \
                (( self.mRadius_grid[ brain_region ] / self.mSkinRadius_normalized )**n) \
                * \
                electrode_diff_legendre[ brain_region ]
            self.mEPot[ skull_region ] += \
                ( \
                self.S(n) \
                * \
                (self.mRadius_grid[ skull_region ]**n) \
                + \
                self.U(n) \
                * \
                (self.mRadius_grid[ skull_region ]**(-1 * (n+1))) \
                ) \
                * \
                electrode_diff_legendre[ skull_region ]
            self.mEPot[ skin_region ]  += \
                ( \
                self.T(n) \
                * \
                (self.mRadius_grid[ skin_region ]**n) \
                + \
                self.W(n) \
                * \
                (self.mRadius_grid[ skin_region ]**(-1 * (n+1))) \
                ) \
                * \
                electrode_diff_legendre[ skin_region ]
            self.mEPot[ buffer_region ]  += \
                ( \
                self.T(n) \
                * \
                (self.mRadius_grid[ buffer_region ]**n) \
                + \
                self.W(n) \
                * \
                (self.mRadius_grid[ buffer_region ]**(-1 * (n+1))) \
                ) \
                * \
                electrode_diff_legendre[ buffer_region ]

        self.mEPot *= summation_factor
        
    @Slot()
    def calcEF( self ):
        self.mEF = np.array( np.gradient( self.mEPot ) )
        self.mEF = -1. * np.linalg.norm( self.mEF, axis=0 )   # we are not interested in the direction

    @Slot(float,float,float)
    def setRadii( self, skin, skull, brain ):
        try:
            self.mSkinRadius  = float(skin)
            self.mSkullRadius = float(skull)
            self.mBrainRadius = float(brain)
            

            outer_radius = 1
            if self.mNormalizeRadii:
                outer_radius = self.mSkinRadius

            self.mSkinRadius_normalized = self.mSkinRadius / outer_radius
            self.mSkullRadius_normalized = self.mSkullRadius / outer_radius
            self.mBrainRadius_normalized = self.mBrainRadius / outer_radius
        except ValueError:
            print("Invalid radii set. Only floating point values are allowed!")

    @Slot(float,float,float)
    def setConductivities( self, skin, skull, brain ):
        try:
            self.mSkinCond  = float(skin)
            self.mSkullCond = float(skull)
            self.mBrainCond = float(brain)
        except ValueError:
            print("Invalid conductivities set. Only floating point values are allowed!")

    @Slot(float, float, int)
    def writeElectrodeCoordinate( self, theta, phi, electrodeNumber=None ):
        assert phi <= 2 * np.pi and phi >= 0 and theta >= 0 and theta <= np.pi

        if  electrodeNumber == None or electrodeNumber >= len(self.mElectrodeCoords):
            self.mElectrodeCoords.append( (theta, phi) )
        else:
            self.mElectrodeCoords[ electrodeNumber - 1 ] = (theta, phi)


    def getData( self ):
        return self.mActiveField 

    def getEPot( self ):
        return self.mEPot

    def getEF( self ):
        return self.mEF

    def activateEF( self ):
        self.mActiveField = self.mEF

    def activateEPot( self ):
        self.mActiveField = self.mEPot

if __name__ == '__main__':
    app    = QApplication( sys.argv )
    window = AppView()
    sys.exit( app.exec_() )
