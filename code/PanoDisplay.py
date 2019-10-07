#Screenshots

#Import sphere (.x file)
from math import pi, sin, cos, tan, atan
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.interval.IntervalGlobal import *
from panda3d.core import *
import sys
from direct.gui.OnscreenText import OnscreenText
import numpy as np
from io import StringIO, BytesIO
import os

from PIL import Image, ImageGrab
import time

import cv2

class PanoDisplay(ShowBase):
    '''
    PanoDisplay class
    
    Description:
        Short: Loads the sphere, applies a texture to it and renders the environment (here, the sphere seen from center, but this could
               be any 3D environment) through a 360 degree fisheye.
        Long: - loads the sphere model (created in Blender; .x format),
              - loads the texture and applies it to the sphere. The sphere has a UV mesh,
              - creates the fisheye setup,
              - has the ability to take screenshots or movies,
              - has the ability to close itself.
              
    Inputs: - Image (.png),
            - window origin,
            - window size.
    
    Outputs:- Graphic window,
            - .png screenshots
    
    Example:
                try:
                    pano = PanoDisplay()    
                    pano.run()
                except SystemExit:
                    pass
    
    '''
    
    
    def __init__(self, filename = None, scale = None, offset = None, win_origin = [0,0], win_size = [1280,720]):
        
        bufferSize = 1024
        
        #Window size and position
        loadPrcFileData("", "win-origin {} {}".format(win_origin[0],win_origin[1]))
        loadPrcFileData("", "win-size {} {}".format(win_size[0],win_size[1]))
        loadPrcFileData("", "undecorated 1")
        
        ShowBase.__init__(self)
        self.textObject = OnscreenText(text = "", pos = (-0.95, 0.9), scale = 0.07)
        
        base.setFrameRateMeter(True)
        # SPHERE
        self.sphere = self.loader.loadModel('res/equirect_sphere.x')
        if filename is None:
            print('No texture or filename given. Loading default texture')
            filename = 'res/test_panodisplay_stretched.png'
        self.filename = filename 
        if type(self.filename) is list:
            self.myTexture  = [self.loader.loadTexture(f) for f in self.filename]#, 'res/stim_mask.png') for f in self.filename]
        else:
            self.myTexture = [self.loader.loadTexture(self.filename)]
        self.iTexture = 0
        self.ts = TextureStage('ts')
        #self.ts.setMode(TextureStage.MDecal)
    #self.ts.setMode(TextureStage.MReplace)
        #
        for t in self.myTexture:
            t.setWrapU(Texture.WMBorderColor)
            t.setWrapV(Texture.WMBorderColor)
            t.setBorderColor(VBase4(1, 1, 1, 1))
        #
#         self.sphere.setTexture(self.ts, self.myTexture[0])
        
        if scale is None:
            self.uScale, self.vScale = 1, 1
        else:
            self.uScale, self.vScale = scale
        self.sphere.setTexScale(self.ts, self.uScale, self.vScale)
        
        if offset is None:
            self.uOffset, self.vOffset = 0, 0
        else:
            self.uOffset, self.vOffset = offset
        
        self.sphere.setTexOffset(self.ts, self.uOffset, self.vOffset)
        
#         bgTexture = self.loader.loadTexture('res/gray.png')
#         self.tsbg = TextureStage('tsbg')
#         self.sphere.setTexture(self.tsbg, bgTexture)
        
#         #self.sphere.setTransparency(TransparencyAttrib.MAlpha, 1)
        
#         diodeTexture = self.loader.loadTexture('res/lena.png', 'res/lena_mask.png')
#         diodeTexture.setWrapU(Texture.WMBorderColor)
#         diodeTexture.setWrapV(Texture.WMBorderColor)
#         diodeTexture.setBorderColor(VBase4(1, 1, 1, 1))
        
#         uScale, vScale = 5, 1
#         self.tsdiode1 = TextureStage('tsdiode1')
#         self.sphere.setTexture(self.tsdiode1, diodeTexture)
#         self.sphere.setTexScale(self.tsdiode1, uScale, vScale)
#         self.sphere.setTexOffset(self.tsdiode1, -4.5, -0)

#         self.tsdiode2 = TextureStage('tsdiode2')
#         self.sphere.setTexture(self.tsdiode2, diodeTexture)
#         self.sphere.setTexScale(self.tsdiode2, uScale, vScale)
#         self.sphere.setTexOffset(self.tsdiode2, 0.5, -0)
        
#         self.tsdiode1.setMode(TextureStage.MDecal)
#         self.tsdiode2.setMode(TextureStage.MDecal)
        #
        #self.uScale = uScale
        #self.vScale = vScale
        #
        #self.x = 0
        #self.y = 0
        #
        #self.xmap = np.zeros((bufferSize,bufferSize), np.float32)
        #self.ymap = np.zeros((bufferSize,bufferSize), np.float32)
        
        self.sphere.reparentTo(self.render)
        self.sphere.setPos(0,0, 0)
        self.sphere.setScale(5,5,5)
        # make the cube map buffer.
        size = bufferSize
        #self.camLens.setNear(0.01) 
        #self.camLens.setFar(100) 
        rig = self.camera.attachNewNode("rig")
        
        self.camera.setPos(0,0,0)
        self.camera.setHpr(0,0,0)
        rig.setPos(0,0,0)
        rig.setHpr(0,340,0)
        
        buffer = self.win.makeCubeMap("test", size, rig)
        assert buffer
        
        # we now get buffer thats going to hold the texture of our new scene
        self.altBuffer = self.win.makeTextureBuffer("env", bufferSize, bufferSize, to_ram = True)
        # now we have to setup a new scene graph to make this scene
        altRender = NodePath("new render")
        
        # altCam on altRender
        altCam = self.makeCamera2d(self.altBuffer)
        altCam.reparentTo(altRender)

        # make fisheye node on altRender
        numVertices = 10000
        fm = FisheyeMaker('card')
        fm.setNumVertices(numVertices)
        fm.setSquareInscribed(True, 1)
        fm.setReflection(True)
        fm.setFov(359.999)
        # set up fisheye
        card = altRender.attachNewNode(fm.generate())
        card.setTexture(buffer.getTexture())
        #altCam.lookAt(card)
        
        # Disable the scene render on the normal 'render' graph.
        self.win.getDisplayRegion(1).setActive(False)

        finalCard = self.loader.loadModel('res/fisheye.egg')
        finalCard.reparentTo(aspect2d)
        finalCard.setTexture(self.altBuffer.getTexture())
        finalCard.setP(90)
       
        def reset():
            self.trackball.node().setPos(0, 0, 0)
            self.trackball.node().setHpr(0, 0, 0)
            rig.setHpr(0,340,0)
            self.sphere.setHpr(0,0,0)
            self.sphere.setTexOffset(self.ts, 0,0) 
        self.accept('m', lambda: self.sphere.setH(self.sphere, 3))
        self.accept('n', lambda: self.sphere.setH(self.sphere, -3))
        self.accept('arrow_down', lambda: self.sphere.setP(self.sphere, -3))
        self.accept('arrow_up', lambda: self.sphere.setP(self.sphere, 3))
        self.accept('arrow_right', lambda: self.sphere.setR(self.sphere, -3))
        self.accept('arrow_left', lambda: self.sphere.setR(self.sphere, 3))
        
        #self.sphere.setTexOffset(self.ts, -1, -0)
        def updateTex(xc,yc):
            tsx,tsy = self.sphere.getTexOffset(self.ts)
            self.sphere.setTexOffset(self.ts, tsx +xc,tsy+ yc)
        self.accept('t', lambda: updateTex(0.2,0))
        self.accept('y', lambda: updateTex(-0.2,0))
        self.accept('g', lambda: updateTex(0,0.2))
        self.accept('h', lambda: updateTex(0,-0.2))
        self.accept('o', lambda: rig.setZ(rig, -0.05))
        self.accept('p', lambda: rig.setZ(rig, 0.05))
        
        self.accept('r', reset)
        
        def quit():
            #taskMgr.remove('Quit')
            #taskMgr.remove('GenerateMappingData')
            self.destroy()
            self.userExit()
            self.finalizeExit()
        self.accept('q', quit) 
        self.taskMgr.add(self.movie_ts, "playTheStim!")
   
        
        def changeFrame():
            self.myTexture = self.loader.loadTexture('res/colored_picture.png')
            self.sphere.setTexture(self.myTexture)
        
        #self.accept('c', changeFrame)
        #self.taskMgr.remove("igLoop")
        #self.taskMgr.add(self.generateMappingData, 'GenerateMappingData')
        #taskMgr.doMethodLater(0.02, quit, 'Quit')
        
    def screenshot(self, filename):
        filename = filename.replace('.png','').replace('flat','pano')
        self.altBuffer.saveScreenshot(filename + '_screenshot.png') 

        
    def generateMappingData(self, task):
        
        x = self.x
        y = self.y
           
        self.sphere.setTexOffset(self.ts, -x, -y)
        
        self.win.setActive(True)
        self.graphicsEngine.renderFrame()
        self.win.setActive(False)
        
        #if self.altBuffer.hasTexture():
        tex = self.altBuffer.getTexture()
        data = tex.getRamImageAs('RGBA')
        if len(data) == 0:
            print('No data in texture ram image -> skipping frame.')
        else:
            im = np.frombuffer(data,np.uint8).reshape((tex.getYSize(),tex.getXSize(),4))
            gray = cv2.cvtColor(im, cv2.COLOR_RGBA2GRAY)
            #Get black pixels coordinates
            #cv2.imwrite('calib/pano/frame_{}_{}.png'.format(x,y),gray)
            idx = np.where(gray<200)
            #print(idx)
            self.xmap[idx], self.ymap[idx] = x, y
        
            self.y += 1
            if self.y == self.vScale:
                self.y = 0
                self.x += 1
                if self.x % 10 == 0:
                    print(self.x)
                if self.x == self.uScale:
                    #write mapping data
                    np.savetxt('res/xmap.txt', self.xmap, fmt='%4d') 
                    np.savetxt('res/ymap.txt', self.ymap, fmt='%4d')
                    print('Task done')
                    return Task.done
        return Task.cont
    
    def movie_ts(self, task):
        self.sphere.setTexture(self.ts,self.myTexture[np.mod(self.iTexture,len(self.myTexture))])
        self.iTexture += 1
        return Task.cont
    
    def movie(self, task):
        self.sphere.setTexture(self.textures[self.textureIndex])
        self.screenshot(self.filenames[self.textureIndex])
        self.textureIndex += 1
        if  self.textureIndex == len(self.textures):
            print('Task done')
            return Task.done
        return Task.cont
        
    def displayCamCoordinates(self, task):
        self.textObject.setText(str(self.sphere.getHpr()) )
        return Task.cont
