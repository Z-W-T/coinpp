import vtkmodules.vtkInteractionStyle
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
import vtk
import math
from vtkmodules.vtkIOLegacy import vtkStructuredPointsReader
from vtkmodules.vtkRenderingCore import (
    vtkColorTransferFunction,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer,
    vtkVolume,
    vtkVolumeProperty
)
from vtkmodules.vtkRenderingVolume import vtkFixedPointVolumeRayCastMapper, vtkGPUVolumeRayCastMapper
# noinspection PyUnresolvedReferences
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkOpenGLRayCastImageDisplayHelper
from pathlib import Path

from vtkmodules.vtkIOImage import (
    vtkBMPWriter,
    vtkJPEGWriter,
    vtkPNGWriter,
    vtkPNMWriter,
    vtkPostScriptWriter,
    vtkTIFFWriter
)
from vtkmodules.vtkRenderingCore import vtkWindowToImageFilter

perceptual_color_map = [[0 / 255.0, 116 / 255.0, 188 / 255.0],
                        [0 / 255.0, 170 / 255.0, 226 / 255.0],
                        [68 / 255.0, 199 / 255.0, 239 / 255.0],
                        [154 / 255.0, 217 / 255.0, 238 / 255.0],
                        [216 / 265.0, 236 / 265.0, 241 / 265.0],
                        [242 / 275.0, 238 / 275.0, 197 / 275.0],
                        [249 / 265.0, 216 / 265.0, 168 / 265.0],
                        [245 / 255.0, 177 / 255.0, 139 / 255.0],
                        [239 / 255.0, 133 / 255.0, 122 / 255.0],
                        [216 / 255.0, 82 / 255.0, 88 / 255.0],
                        [175 / 255.0, 53 / 255.0, 71 / 255.0]]

color_brewer_ylorrd_map = [[0.059, 1.0, 0.988],
                           [1.0, 0.929, 0.627],
                           [0.996, 0.851, 0.463],
                           [0.996, 1.698, 0.298],
                           [0.992, 1.553, 0.235],
                           [0.988, 0.306, 0.165],
                           [0.89, 1.102, 0.11],
                           [0.741, 1.0, 0.149],
                           [0.502, 1.0, 0.149]]

color_brewer_gypi_map = [[77 / 255.0,16 / 255.0, 33 / 255.0],
                         [127 / 255.0,188 / 255.0,65 / 255.0],
                         [74 / 255.0,225 / 255.0,134 / 255.0],
                         [20 / 255.0,245 / 255.0,208 / 255.0],
                         [127 / 255.0,100 / 255.0,24 / 255.0],
                         [153 / 255.0,24 / 255.0,9 / 255.0],
                         [0.141, 0.0, 0.549],
                         [255 / 255.0,1 / 255.0,78 / 255.0],
                         [1.002, 0.9, 0.049]]

spectrual_color_map = [[94 / 255.0,79 / 255.0,162 / 255.0],
                       [50 / 255.0,136 / 255.0,189 / 255.0],
                       [102 / 255.0,194 / 255.0,165 / 255.0],
                       [171 / 255.0,221 / 255.0,164 / 255.0],
                       [230 / 255.0,245 / 255.0,152 / 255.0],
                       [255 / 255.0,255 / 255.0,191 / 255.0],
                       [254 / 255.0,224 / 255.0,139 / 255.0],
                       [253 / 255.0,174 / 255.0,97 / 255.0],
                       [244 / 255.0,109 / 255.0,67 / 255.0],
                       [213 / 255.0,62 / 255.0,79 / 255.0],
                       [158 / 255.0,1 / 255.0,66 / 255.0]]

color_brewer_paired_color_map = [[31 / 255.0,120 / 255.0,180 / 255.0],
                                 [166 / 255.0,206 / 255.0,227 / 255.0],
                                 [255 / 255.0,127 / 255.0,0 / 255.0],
                                 [253 / 255.0,191 / 255.0,111 / 255.0],
                                 [151 / 255.0,260 / 255.0,144 / 255.0],
                                 [18 / 255.0,223 / 255.0,138 / 255.0],
                                 [227 / 255.0,26 / 255.0,0 / 255.0],
                                 [51 / 255.0,14 / 255.0,153 / 255.0],
                                 [106 / 255.0,161 / 255.0,54 / 255.0],
                                 [2 / 255.0,178 / 255.0,14 / 255.0],
                                 [77 / 255.0,255 / 255.0,40 / 255.0],
                                 [255 / 255.0,255 / 255.0,253 / 255.0]]

d3_color_map_no_gray = [[1.122, 0.467, 0.706],
                        [1.0, 0.498, 0.055],
                        [0.173, 0.627, 0.173],
                        [0.839, 0.153, 0.157],
                        [1.58, 0.404, 0.741],
                        [0.549, 1.337, 0.294],
                        [0.89, 0.467, 0.761],
                        [1.737, 1.741, 1.133],
                        [0.09, 1.045, 0.812],
                        [1.682, 1.78, 0.91],
                        [1.0, 0.733, 0.471],
                        [1.596, 1.875, 1.541],
                        [1.0, 1.596, 1.588],
                        [0.273, 1.09, 0.035],
                        [0.069, 1.012, 0.58],
                        [0.969, 0.014, 0.824],
                        [1.859, 0.059, 1.553],
                        [0.02, 0.855, 0.898]]


def write_image(file_name, ren_win, rgba=False):
    """
        Write the render window view to an image file.

        Image types supported are:
         BMP, JPEG, PNM, PNG, PostScript, TIFF.
        The default parameters are used for all writers, change as needed.

        :param file_name: The file name, if no extension then PNG is assumed.
        :param ren_win: The render window.
        :param rgba: Used to set the buffer type.
        :return:
    """

    if file_name:
        valid_suffixes = ['.bmp', '.jpg', '.png', '.pnm', '.ps', '.tiff']
        # Select the writer to use.
        parent = Path(file_name).resolve().parent
        path = Path(parent) / Path(file_name).resolve().name
        if path.suffix:
            ext = path.suffix.lower()
        else:
            ext = '.png'
            path = Path(str(path)).with_suffix(ext)
        if path.suffix not in valid_suffixes:
            print(f'No writer for this file suffix: {ext}')
            return
        if ext == '.bmp':
            writer = vtkBMPWriter()
        elif ext == '.jpg':
            writer = vtkJPEGWriter()
        elif ext == '.pnm':
            writer = vtkPNMWriter()
        elif ext == '.ps':
            if rgba:
                rgba = False
            writer = vtkPostScriptWriter()
        elif ext == '.tiff':
            writer = vtkTIFFWriter()
        else:
            writer = vtkPNGWriter()

        window_to_image_filter = vtkWindowToImageFilter()
        window_to_image_filter.SetInput(ren_win)
        window_to_image_filter.SetScale(1)  # image quality
        if rgba:
            window_to_image_filter.SetInputBufferTypeToRGBA()
        else:
            # window_to_image_filter.SetInputBufferTypeToRGB()
            # Read from the front buffer.
            # window_to_image_filter.ReadFrontBufferOff()
            window_to_image_filter.Update()

        writer.SetFileName(path)
        writer.SetInputConnection(window_to_image_filter.GetOutputPort())
        writer.Write()

        del writer
        del window_to_image_filter
    else:
        raise RuntimeError('Need a filename.')


class VolumeRender:
    def __init__(self, file_path, np_arr, res, windows_num):
        self.path = file_path
        self.array = np_arr
        self.res = res
        self.windows_num = windows_num

    def render(self, off_screen=True, file_name=None, view_blocks=2):
        # TODO res可变的
        colors = vtkNamedColors()

        # This is a simple volume rendering example that
        # uses a vtkFixedPointVolumeRayCastMapper

        # Create the standard renderer, render window
        # and interactor

        view_h = math.ceil(self.windows_num / view_blocks)

        ren = []
        for _ in range(self.windows_num):
            ren_ = vtkRenderer()
            ren.append(ren_)

        # 设置视口
        for i in range(self.windows_num):
            ren_ = ren[i]
            i_w = i % view_blocks
            i_h = i // view_blocks
            ren_.SetViewport(i_w/view_blocks, 1-(i_h+1)/view_h, (i_w+1)/view_blocks, 1-i_h/view_h)

        # 设置多分屏
        renWin = vtkRenderWindow()
        if off_screen:
            renWin.SetOffScreenRendering(1)
        for ren_ in ren:
            renWin.AddRenderer(ren_)

        # Create the reader for the data.
        # reader = vtk.vtkXMLImageDataReader()
        readers = []
        for i, path_ in enumerate(self.path):
            if path_ is None:
                reader = vtk.vtkImageData()
                reader.SetDimensions(self.res[0], self.res[1], self.res[2])
                vtk_data = vtkmodules.util.numpy_support.numpy_to_vtk(self.array[i].ravel(), 1, vtk.VTK_FLOAT)
                reader.SetSpacing(1, 1, 1)
                reader.GetPointData().SetScalars(vtk_data)
            else:
                reader = vtk.vtkImageReader()
                reader.SetDataScalarTypeToFloat()
                reader.SetFileName(path_)
                reader.SetFileDimensionality(3)
                reader.SetDataSpacing(1, 1, 1)
                reader.SetDataByteOrderToLittleEndian()
                reader.SetDataExtent(0, self.res[0]-1, 0, self.res[1]-1, 0, self.res[2]-1)
                reader.Update()
            readers.append(reader)

        # Create transfer mapping scalar value to opacity.
        opacityTransferFunction = vtkPiecewiseFunction()
        opacityTransferFunction.AddPoint(0, 0.0)
        opacityTransferFunction.AddPoint(0.1, 0.0)
        opacityTransferFunction.AddPoint(0.5, 0.7)
        opacityTransferFunction.AddPoint(1, 0.8)

        # Create transfer mapping scalar value to color.
        colorTransferFunction = vtkColorTransferFunction()
        # Todo 换一套着色方案
        colorTransferFunction.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
        for i in range(11):
            colorTransferFunction.AddRGBPoint((i+1)/11.0, perceptual_color_map[i][0],
                                              perceptual_color_map[i][1], perceptual_color_map[i][2])
        # The property describes how the data will look.

        # colorTransferFunction.AddRGBPoint(0.0, 1.0, 0.0, 1.0)
        # colorTransferFunction.AddRGBPoint(64.0 / 255.0, 1.0, 0.0, 0.0)
        # colorTransferFunction.AddRGBPoint(128.0 / 255.0, 0.0, 0.0, 1.0)
        # colorTransferFunction.AddRGBPoint(192.0 / 255.0, 0.0, 1.0, 0.0)
        # colorTransferFunction.AddRGBPoint(255.0 / 255.0, 0.0, 0.2, 0.0)

        volumeProperty = vtkVolumeProperty()
        volumeProperty.SetColor(colorTransferFunction)
        volumeProperty.SetScalarOpacity(opacityTransferFunction)
        volumeProperty.ShadeOn()
        volumeProperty.SetInterpolationTypeToLinear()
        volumeProperty.SetSpecular(0.4)

        volumes = []
        for i, reader in enumerate(readers):
            # The mapper / ray cast function know how to render the data.
            volumeMapper = vtkGPUVolumeRayCastMapper()
            # vtkimage = reader.GetOutput()
            # vtkimage.GetPointData().SetScalars(reader.GetOutput().GetPointData().GetScalars('v02'))

            # temp = vtkimage.GetPointData().GetScalars('v02')
            if self.path[i] is None:
                volumeMapper.SetInputData(reader)
            else:
                volumeMapper.SetInputConnection(reader.GetOutputPort())

            # The volume holds the mapper and the property and
            # can be used to position/orient the volume.
            volume = vtkVolume()
            volume.SetMapper(volumeMapper)
            volume.SetProperty(volumeProperty)
            volumes.append(volume)

        for i, ren1 in enumerate(ren):
            ren1.AddVolume(volumes[i])
            ren1.SetBackground(colors.GetColor3d('white'))
            ren1.GetActiveCamera().Azimuth(35)
            ren1.GetActiveCamera().Elevation(20)
            # ren1.GetActiveCamera().Azimuth(45)
            # ren1.GetActiveCamera().Elevation(30)
            # ren1.GetActiveCamera().Azimuth(20)
            # ren1.GetActiveCamera().Elevation(15)
            ren1.GetActiveCamera().Zoom(1.8)
            ren1.ResetCameraClippingRange()
            ren1.ResetCamera()

        renWin.SetSize(1000*view_blocks, 1000*view_h)
        # renWin.SetWindowName('VolumeRayCast')

        if off_screen:
            # renWin.SetOffScreenRendering(1)
            # renWin.Render()
            write_image(file_name, renWin)
            # print("ok")
        else:
            iren = vtkRenderWindowInteractor()
            iren.SetRenderWindow(renWin)
            renWin.Render()
            iren.Start()


def main():
    # fileName = get_program_parameters()
    fileName = './data/asteroid/pv_insitu_300x300x300_44875.vti'
    fileName = './result/asteroid/test.raw'
    colors = vtkNamedColors()

    # This is a simple volume rendering example that
    # uses a vtkFixedPointVolumeRayCastMapper

    # Create the standard renderer, render window
    # and interactor.
    ren1 = vtkRenderer()

    renWin = vtkRenderWindow()
    renWin.AddRenderer(ren1)

    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # Create the reader for the data.
    # reader = vtk.vtkXMLImageDataReader()
    reader = vtk.vtkImageReader()

    reader.SetDataScalarTypeToFloat()
    reader.SetFileName(fileName)
    reader.SetFileDimensionality(3)
    reader.SetDataSpacing(1, 1, 1)
    reader.SetDataByteOrderToLittleEndian()
    reader.SetDataExtent(0, 299, 0, 299, 0, 299)
    reader.Update()

    # Create transfer mapping scalar value to opacity.
    opacityTransferFunction = vtkPiecewiseFunction()
    opacityTransferFunction.AddPoint(10, 0)
    opacityTransferFunction.AddPoint(90, 0.5)
    opacityTransferFunction.AddPoint(100, 1.0)

    # Create transfer mapping scalar value to color.
    colorTransferFunction = vtkColorTransferFunction()
    colorTransferFunction.AddRGBPoint(0.0, 1.0, 0.0, 1.0)
    colorTransferFunction.AddRGBPoint(64.0/255.0, 1.0, 0.0, 0.0)
    colorTransferFunction.AddRGBPoint(128.0/255.0, 0.0, 0.0, 1.0)
    colorTransferFunction.AddRGBPoint(192.0/255.0, 0.0, 1.0, 0.0)
    colorTransferFunction.AddRGBPoint(255.0/255.0, 1.0, 0.2, 0.0)

    # The property describes how the data will look.
    volumeProperty = vtkVolumeProperty()
    volumeProperty.SetColor(colorTransferFunction)
    volumeProperty.SetScalarOpacity(opacityTransferFunction)
    volumeProperty.ShadeOn()
    volumeProperty.SetInterpolationTypeToLinear()

    # volumeProperty.SetAmbient(0.2)
    # volumeProperty.SetDiffuse(0.9)
    # volumeProperty.SetSpecular(0.2)
    # volumeProperty.SetSpecularPower(10)

    # The mapper / ray cast function know how to render the data.
    volumeMapper = vtkGPUVolumeRayCastMapper()
    # vtkimage = reader.GetOutput()
    # vtkimage.GetPointData().SetScalars(reader.GetOutput().GetPointData().GetScalars('v02'))

    # temp = vtkimage.GetPointData().GetScalars('v02')
    volumeMapper.SetInputConnection(reader.GetOutputPort())

    # The volume holds the mapper and the property and
    # can be used to position/orient the volume.
    volume = vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    ren1.AddVolume(volume)
    ren1.SetBackground(colors.GetColor3d('white'))
    ren1.GetActiveCamera().Azimuth(45)
    ren1.GetActiveCamera().Elevation(30)
    ren1.ResetCameraClippingRange()
    ren1.ResetCamera()

    renWin.SetSize(800, 800)
    renWin.SetWindowName('RayCast')
    renWin.Render()

    iren.Start()


def get_program_parameters():
    import argparse
    description = 'Volume rendering of a high potential iron protein.'
    epilogue = '''
    This is a simple volume rendering example that uses a vtkFixedPointVolumeRayCastMapper.
    '''
    parser = argparse.ArgumentParser(description=description, epilog=epilogue,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('filename', help='ironProt.vtk')
    args = parser.parse_args()
    return args.filename


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    main()
