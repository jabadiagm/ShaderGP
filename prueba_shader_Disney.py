import OpenGL.GLUT as glut
import OpenGL.GLU as glu
import OpenGL.GL as gl

import sys
import numpy as np
import ctypes as ctypes
import math as math
from PIL import Image as Image
from PIL import ImageOps as ImageOps
import struct
import time as time

name = 'GLSL Disney Shader Demo'
model_rotx = 0.0
model_roty = 0.0
model_rotz = 0.0
view_rotx = 0.0
view_roty = 0.0
view_rotz = 0.0

light1Type = 1
light2Type = 1
light3Type = 1
light4Type = 1
light5Type = 0

CAMERA_RADIUS = 10

multiSampling = False

VerticesCubo = np.zeros(24, [("position", np.float32, 3)])
VerticesCubo["position"] = [[ 1, 1, 1], [-1, 1, 1], [-1,-1, 1], [ 1,-1, 1],
                            [ 1, 1, 1], [ 1,-1, 1], [ 1,-1,-1], [ 1, 1,-1],
                            [ 1, 1, 1], [ 1, 1,-1], [-1, 1,-1], [-1, 1, 1],
                            [-1, 1, 1], [-1, 1,-1], [-1,-1,-1], [-1,-1, 1],
                            [-1,-1,-1], [ 1,-1,-1], [ 1,-1, 1], [-1,-1, 1],
                            [ 1,-1,-1], [-1,-1,-1], [-1, 1,-1], [ 1, 1,-1]]  

IndicesCubo = np.array([0,1,2, 2,3,0,  4,5,6, 6,7,4,  8,9,10, 10,11,8, 
           12,13,14, 14,15,12,  16,17,18, 18,19,16,  20,21,22, 22,23,20], dtype=np.uint32)     

NormalesCubo = np.zeros(24, [("normal", np.float32, 3)])
NormalesCubo["normal"] = [[ 0, 0, 1], [ 0, 0, 1], [ 0, 0, 1], [ 0, 0, 1],
                          [ 1, 0, 0], [ 1, 0, 0], [ 1, 0, 0], [ 1, 0, 0],
                          [ 0, 1, 0], [ 0, 1, 0], [ 0, 1, 0], [ 0, 1, 0],
                          [-1, 0, 0], [-1, 0, 0], [-1, 0, 0], [-1, 0, 0],
                          [ 0,-1, 0], [ 0,-1, 0], [ 0,-1, 0], [ 0,-1, 0],
                          [ 0, 0,-1], [ 0, 0,-1], [ 0, 0,-1], [ 0, 0,-1]]   

CoordTexCubo = np.zeros(24, [("tex", np.float32, 2)])
CoordTexCubo["tex"] = [[1, 0], [0, 0], [0, 1], [1, 1],
                       [0, 0], [0, 1], [1, 1], [1, 0],
                       [1, 1], [1, 0], [0, 0], [0, 1],
                       [1, 0], [0, 0], [0, 1], [1, 1],
                       [0, 1], [1, 1], [1, 0], [0, 0],
                       [0, 1], [1, 1], [1, 0], [0, 0]]  


buffer_offset = ctypes.c_void_p
float_size = ctypes.sizeof(ctypes.c_float)


#CAMERA_FOVY = 45.0
CAMERA_FOVY = 20.0
CAMERA_ZFAR = 100.0
CAMERA_ZNEAR = 0.1

DOLLY_MAX = 10.0
DOLLY_MIN = 2.5

MOUSE_ORBIT_SPEED = 0.30
MOUSE_DOLLY_SPEED = 0.02
MOUSE_TRACK_SPEED = 0.005

SPOT_INNER_CONE = 10.0
SPOT_OUTER_CONE = 15.0

LIGHT_RADIUS = 10.0

#Globals.
g_nullTexture = 0
g_disneyShaderProgram = 0
g_vertexBuffer = 0
g_vertexIndexBuffer = 0
g_disableColorMapTexture = False
g_colorMapTexture = 0
g_normalMapTexture = 0
g_roughnessMapTexture = 0
g_metalnessMapTexture = 0
g_windowWidth = 500
g_windowHeight = 500
g_startTime = time.time()
g_frames = 0

g_cube = np.zeros(24, [("Vertex", np.float32, 12)])
g_cube["Vertex"] = [
    # Positive Z Face
    [ -1.0, -1.0,  1.0,  0.0, 0.0,  0.0, 0.0, 1.0,  0.0, 0.0, 0.0, 0.0],
    [  1.0, -1.0,  1.0,  1.0, 0.0,  0.0, 0.0, 1.0,  0.0, 0.0, 0.0, 0.0],
    [  1.0,  1.0,  1.0,  1.0, 1.0,  0.0, 0.0, 1.0,  0.0, 0.0, 0.0, 0.0],
    [ -1.0,  1.0,  1.0,  0.0, 1.0,  0.0, 0.0, 1.0,  0.0, 0.0, 0.0, 0.0],

    # Negative Z Face
    [  1.0, -1.0, -1.0,  0.0, 0.0,  0.0, 0.0, -1.0,  0.0, 0.0, 0.0, 0.0],
    [ -1.0, -1.0, -1.0,  1.0, 0.0,  0.0, 0.0, -1.0,  0.0, 0.0, 0.0, 0.0],
    [ -1.0,  1.0, -1.0,  1.0, 1.0,  0.0, 0.0, -1.0,  0.0, 0.0, 0.0, 0.0],
    [  1.0,  1.0, -1.0,  0.0, 1.0,  0.0, 0.0, -1.0,  0.0, 0.0, 0.0, 0.0],

    # Positive Y Face
    [ -1.0,  1.0,  1.0,  0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 0.0, 0.0],
    [  1.0,  1.0,  1.0,  1.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 0.0, 0.0],
    [  1.0,  1.0, -1.0,  1.0, 1.0,  0.0, 1.0, 0.0,  0.0, 0.0, 0.0, 0.0],
    [ -1.0,  1.0, -1.0,  0.0, 1.0,  0.0, 1.0, 0.0,  0.0, 0.0, 0.0, 0.0],

    # Negative Y Face
    [ -1.0, -1.0, -1.0,  0.0, 0.0,  0.0, -1.0, 0.0,  0.0, 0.0, 0.0, 0.0],
    [  1.0, -1.0, -1.0,  1.0, 0.0,  0.0, -1.0, 0.0,  0.0, 0.0, 0.0, 0.0],
    [  1.0, -1.0,  1.0,  1.0, 1.0,  0.0, -1.0, 0.0,  0.0, 0.0, 0.0, 0.0],
    [ -1.0, -1.0,  1.0,  0.0, 1.0,  0.0, -1.0, 0.0,  0.0, 0.0, 0.0, 0.0],

    # Positive X Face
    [  1.0, -1.0,  1.0,  0.0, 0.0,  1.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0],
    [  1.0, -1.0, -1.0,  1.0, 0.0,  1.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0],
    [  1.0,  1.0, -1.0,  1.0, 1.0,  1.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0],
    [  1.0,  1.0,  1.0,  0.0, 1.0,  1.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0],

    # Negative X Face
    [ -1.0, -1.0, -1.0,  0.0, 0.0,  -1.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0],
    [ -1.0, -1.0,  1.0,  1.0, 0.0,  -1.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0],
    [ -1.0,  1.0,  1.0,  1.0, 1.0,  -1.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0],
    [ -1.0,  1.0, -1.0,  0.0, 1.0,  -1.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0]    
    ] 

g_cube_index = np.array ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],np.uint32)

vertices = np.zeros(3, [("position", np.float32, 2)])
vertices["position"] = [[ 1, 1], [-1, 1], [-1,-1]] 
verticesbyte=bytearray(vertices)
indices = (0, 1, 2)

file = open("vertex.gsl","r")
vertex = file.read()
file.close()

file = open("fragment.gsl","r")
fragment = file.read()
file.close()

shaderDisney=([vertex], [fragment])


def main():
    global g_colorMapTexture
    global g_normalMapTexture
    global g_roughnessMapTexture
    global g_metalnessMapTexture
    global g_nullTexture
    global g_dirNormalMappingShaderProgram
    global g_pointNormalMappingShaderProgram
    global g_spotNormalMappingShaderProgram
    global g_disneyShaderProgram
   
    
    glut.glutInit(sys.argv)
    
    if not multiSampling:
        glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGB | glut.GLUT_DEPTH)
    else:
        glut.glutSetOption(glut.GLUT_MULTISAMPLE, 2)
        glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGB | glut.GLUT_DEPTH | glut.GLUT_MULTISAMPLE)
    
    glut.glutInitWindowPosition(0, 0)
    glut.glutInitWindowSize(g_windowWidth, g_windowHeight)
    glut.glutCreateWindow(name)
    
    #glut.glutDisplayFunc(display)
    glut.glutDisplayFunc(RenderFrame)
    glut.glutKeyboardFunc(key)
    glut.glutReshapeFunc(reshape)
    glut.glutSpecialFunc(special)
    glut.glutVisibilityFunc(visible)   
    
    GL_MAX_VARYING_FLOATS = gl.glGetIntegerv(gl.GL_MAX_VARYING_FLOATS) 
    print("GL_MAX_VARYING_FLOATS=",GL_MAX_VARYING_FLOATS)
    
    #InitApp

    # Load the GLSL normal mapping shaders
    g_disneyShaderProgram=LoadShaderProgram(shaderDisney)
    print("g_disneyShaderProgram=",g_disneyShaderProgram)
    
    
    # Load the textures
    g_colorMapTexture = LoadTexture("Metal015_2K_Color.jpg")
    g_normalMapTexture = LoadTexture("Metal015_2K_Normal.jpg")
    g_roughnessMapTexture = LoadTexture("Metal015_2K_Roughness.jpg")
    g_metalnessMapTexture = LoadTexture("Metal015_2K_Metalness.jpg")
    g_nullTexture =  CreateNullTexture(2, 2)
    
    # Initialize the cube model
    InitCube()
    
    #Setup initial rendering states
    
    gl.glEnable(gl.GL_TEXTURE_2D)
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glEnable(gl.GL_CULL_FACE)
    gl.glEnable(gl.GL_LIGHTING)
    gl.glEnable(gl.GL_LIGHT0)   
    
    
    #while(True):
    #    RenderFrame()
    #    UpdateFrameRate()

    
    glut.glutMainLoop()
    return
    
class Material():
    baseColor = [.82, .67, .16]
    metallic = 0.0
    subsurface = 0
    specular = 0.0
    roughness = 0.0
    specularTint = 0
    anisotropic = 0
    sheen = 0
    sheenTint = .5
    clearcoat = 0
    clearcoatGloss = 0
    
    colorMapTexture = 0
    normalMapTexture = 0
    roughnessMapTexture = 0
    metalnessMapTexture = 0
    


        
def InitCube():
    global g_vertexBuffer
    global g_vertexIndexBuffer
    global g_cube
    
    for i in range(0, 24, 4):
        pVertex1 = g_cube[i][0]
        pVertex2 = g_cube[i + 1][0]
        pVertex3 = g_cube[i + 2][0]
        pVertex4 = g_cube[i + 3][0]
        
        tangent = CalcTangentVector(pVertex1[0:3], pVertex2[0:3], pVertex4[0:3], pVertex1[3:5], pVertex2[3:5], pVertex4[3:5], pVertex1[5:8])
        pVertex1[8:12] = tangent
        pVertex2[8:12] = tangent
        pVertex3[8:12] = tangent
        pVertex4[8:12] = tangent
        
    # Store the cube's geometry in a Vertex Buffer Object
    g_vertexBuffer = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, g_vertexBuffer)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, 24 * 12 * 4, g_cube, gl.GL_STATIC_DRAW)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
    
    g_vertexIndexBuffer = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, g_vertexIndexBuffer)
    gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, 4 * len(g_cube_index), g_cube_index, gl.GL_STATIC_DRAW)     
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
    

def CalcTangentVector(pos1, pos2, pos3, texCoord1, texCoord2, texCoord3, normal):
    # Given the 3 vertices (position and texture coordinates) of a triangle
    # calculate and return the triangle's tangent vector.
    
    #Create 2 vectors in object space
    # edge1 is the vector from vertex positions pos1 to pos2
    # edge2 is the vector from vertex positions pos1 to pos3    
    edge1 = pos2 - pos1
    edge2 = pos3 - pos1
    
    edge1=normalize(edge1)
    edge2=normalize(edge2)
    
    # Create 2 vectors in tangent (texture) space that point in the same
    # direction as edge1 and edge2 (in object space)  
    # texEdge1 is the vector from texture coordinates texCoord1 to texCoord2
    # texEdge2 is the vector from texture coordinates texCoord1 to texCoord3
    texEdge1 = texCoord2 - texCoord1
    texEdge2 = texCoord3 - texCoord1
    
    texEdge1=normalize(texEdge1)
    texEdge2=normalize(texEdge2)

    # These 2 sets of vectors form the following system of equations:
    #
    #  edge1 = (texEdge1.x * tangent) + (texEdge1.y * bitangent)
    #  edge2 = (texEdge2.x * tangent) + (texEdge2.y * bitangent)
    #
    # Using matrix notation this system looks like:
    #
    #  [ edge1 ]     [ texEdge1.x  texEdge1.y ]  [ tangent   ]
    #  [       ]  =  [                        ]  [           ]
    #  [ edge2 ]     [ texEdge2.x  texEdge2.y ]  [ bitangent ]
    #
    # The solution is:
    #
    #  [ tangent   ]        1     [ texEdge2.y  -texEdge1.y ]  [ edge1 ]
    #  [           ]  =  -------  [                         ]  [       ]
    #  [ bitangent ]      det A   [-texEdge2.x   texEdge1.x ]  [ edge2 ]
    #
    #  where:
    #        [ texEdge1.x  texEdge1.y ]
    #    A = [                        ]
    #        [ texEdge2.x  texEdge2.y ]
    #
    #    det A = (texEdge1.x * texEdge2.y) - (texEdge1.y * texEdge2.x)
    #
    # From this solution the tangent space basis vectors are:
    #
    #    tangent = (1 / det A) * ( texEdge2.y * edge1 - texEdge1.y * edge2)
    #  bitangent = (1 / det A) * (-texEdge2.x * edge1 + texEdge1.x * edge2)
    #     normal = cross(tangent, bitangent)

    n=np.array([0, 0, 0],np.float32)
    n[0:3] = normal[0:3]

    det = (texEdge1[0] * texEdge2[1]) - (texEdge1[1] * texEdge2[0])
    
    if abs(det) < 1e-5:
        t=np.array([1,0,0],np.float32)
        b=np.array([0,1,0],np.float32)
    else:
        det = 1.0 / det

        t=np.array([0,0,0],np.float32)
        b=np.array([0,0,0],np.float32)
        
        t[0] = (texEdge2[1] * edge1[0] - texEdge1[1] * edge2[0]) * det
        t[1] = (texEdge2[1] * edge1[1] - texEdge1[1] * edge2[1]) * det
        t[2] = (texEdge2[1] * edge1[2] - texEdge1[1] * edge2[2]) * det

        b[0] = (-texEdge2[0] * edge1[0] + texEdge1[0] * edge2[0]) * det
        b[1] = (-texEdge2[0] * edge1[1] + texEdge1[0] * edge2[1]) * det
        b[2] = (-texEdge2[0] * edge1[2] + texEdge1[0] * edge2[2]) * det        
        
        t = normalize(t)
        b = normalize(b)
        
    # Calculate the handedness of the local tangent space.
    # The bitangent vector is the cross product between the triangle face
    # normal vector and the calculated tangent vector. The resulting bitangent
    # vector should be the same as the bitangent vector calculated from the
    # set of linear equations above. If they point in different directions
    # then we need to invert the cross product calculated bitangent vector. We
    # store this scalar multiplier in the tangent vector's 'w' component so
    # that the correct bitangent vector can be generated in the normal mapping
    # shader's vertex shader

    bitangent = np.cross(n, t)

    if np.dot(bitangent, b) < 0:
        handedness = -1.0
    else:
        handedness = 1.0
   
    tangent = np.array([t[0], t[1], t[2], handedness], np.float32)
    return tangent
    
    
def normalize(vector):
    #divides a numpy vector by its module
    return vector/np.linalg.norm(vector)

def CreateNullTexture(width, height):
    # Create an empty white texture. This texture is applied to models
    # that don't have any texture maps. This trick allows the same shader to
    # be used to draw the model with and without textures applied.    
    
    pitch = ((width * 32 + 31) & bit_not(31,8)) >> 3 # align to 4-byte boundaries
    pixels = np.full ( pitch * height, 255,np.uint8)
    texture = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture)

    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
    
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, width, height, 0, gl.GL_BGRA, gl.GL_UNSIGNED_BYTE, pixels)
    
    return texture

def bit_not(n, numbits=8):
    return (1 << numbits) - 1 - n

def LoadTexture(fileName):
    return LoadTexture2(fileName, gl.GL_LINEAR, gl.GL_LINEAR_MIPMAP_LINEAR,
        gl.GL_REPEAT, gl.GL_REPEAT)

def LoadTexture2(fileName, magFilter, minFilter, wrapS, wrapT):
    img = Image.open(fileName)
    img = ImageOps.flip(img)
    #img_data = np.array(list(img.getdata()), np.int8)
    img_data =np.asarray( img, dtype="int8")
    id = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, id)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, magFilter)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, minFilter)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, wrapS)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, wrapT)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_GENERATE_MIPMAP, gl.GL_TRUE)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, img.size[0],img.size[1], 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, img_data)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    return id


def LoadShaderProgram(shaderProgram):
    vertShader = CompileShader(gl.GL_VERTEX_SHADER, shaderProgram[0])
    fragShader = CompileShader(gl.GL_FRAGMENT_SHADER, shaderProgram[1])
    program = LinkShaders(vertShader, fragShader)
    return program

def CompileShader(type, source):
    # Compiles the shader given it's source code. Returns the shader object.
    # 'type' is either GL_VERTEX_SHADER or GL_FRAGMENT_SHADER.
    shader = gl.glCreateShader(type)
    if shader:
        gl.glShaderSource(shader, source)
        gl.glCompileShader(shader)
        compiled=gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS)
        if not compiled:
            report = gl.glGetShaderInfoLog (shader)
            print("Report=",report)
            raise Exception("Compile error") 
    return shader
    
def LinkShaders(vertShader, fragShader):
    # Links the compiled vertex and/or fragment shaders into an executable
    # shader program. Returns the executable shader object. If the shaders
    # failed to link into an executable shader program, then a std::string
    # object is thrown containing the info log.    
    program = gl.glCreateProgram()
    if program:
        gl.glAttachShader(program, vertShader)
        gl.glAttachShader(program, fragShader)
        gl.glLinkProgram(program)
        linked=gl.glGetProgramiv(program, gl.GL_LINK_STATUS)
        if not linked:
            raise Exception("Linker error")   
        # Mark the two attached shaders for deletion. These two shaders aren't
        # deleted right now because both are already attached to a shader
        # program. When the shader program is deleted these two shaders will
        # be automatically detached and deleted.            
        gl.glDeleteShader(vertShader)
        gl.glDeleteShader(fragShader)
    return program

def cosd(degrees):
    return math.cos(degrees * math.pi / 180)

def sind(degrees):
    return math.sin(degrees * math.pi / 180)


def key(k, x, y):
    global model_rotx
    global model_roty
    global model_rotz
    global light1Type
    global light2Type
    global g_disableColorMapTexture
    if k == b'z':
        model_rotz += 5.0
    elif k == b'Z':
        model_rotz -= 5.0
    elif k == b'x':
        model_rotx += 5.0
    elif k == b'X':
        model_rotx -= 5.0
    elif k == b'y':
        model_roty += 1.0
    elif k == b'Y':
        model_roty -= 1.0
    elif k == b't':
        g_disableColorMapTexture = not g_disableColorMapTexture
    elif k == b'1':
        light1Type = (light1Type + 1) %3
    elif k == b'2':
        light2Type = (light2Type + 1) %3
    elif ord(k) == 27: # Escape
        sys.exit(0)
    else:
        return
    glut.glutPostRedisplay()
    
# change view angle
def special(k, x, y):
    global view_rotx, view_roty
    
    if k == glut.GLUT_KEY_UP:
        view_roty += 2.0
    elif k == glut.GLUT_KEY_DOWN:
        view_roty -= 2.0
    elif k == glut.GLUT_KEY_LEFT:
        view_rotx -= 5.0
    elif k == glut.GLUT_KEY_RIGHT:
        view_rotx += 5.0
    else:
        return
    print("view_rotx=",view_rotx,"view_roty=",view_roty)
    glut.glutPostRedisplay()

# new window size or exposure
def reshape(width, height):
    global g_windowWidth
    global g_windowHeight
    g_windowWidth = width
    g_windowHeight = height

def visible(vis):
    if vis == glut.GLUT_VISIBLE:
        glut.glutIdleFunc(idle)
    else:
        glut.glutIdleFunc(None)  

def idle():
    UpdateFrameRate()
    glut.glutPostRedisplay()
    

def UpdateFrameRate():
    global g_startTime
    global g_frames
    if time.time() - g_startTime > 1.0:
        g_startTime = time.time()
        print("FPS = ",g_frames)
        g_frames = 0
    
def RenderCube():
    
    gl.glActiveTexture(gl.GL_TEXTURE3)
    gl.glEnable(gl.GL_TEXTURE_2D)
    gl.glBindTexture(gl.GL_TEXTURE_2D, g_metalnessMapTexture)    
    
    gl.glActiveTexture(gl.GL_TEXTURE2)
    gl.glEnable(gl.GL_TEXTURE_2D)
    gl.glBindTexture(gl.GL_TEXTURE_2D, g_roughnessMapTexture)    
    
    gl.glActiveTexture(gl.GL_TEXTURE1)
    gl.glEnable(gl.GL_TEXTURE_2D)
    gl.glBindTexture(gl.GL_TEXTURE_2D, g_normalMapTexture)
    
    gl.glActiveTexture(gl.GL_TEXTURE0)
    gl.glEnable(gl.GL_TEXTURE_2D)
    if g_disableColorMapTexture:
        gl.glBindTexture(gl.GL_TEXTURE_2D, g_nullTexture)
    else:
        gl.glBindTexture(gl.GL_TEXTURE_2D, g_colorMapTexture)
    
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, g_vertexBuffer)
    
    gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
    gl.glVertexPointer(3, gl.GL_FLOAT, g_cube.strides[0], ctypes.c_void_p(0))
    
    gl.glClientActiveTexture(gl.GL_TEXTURE0);
    gl.glEnableClientState(gl.GL_TEXTURE_COORD_ARRAY);
    gl.glTexCoordPointer(2, gl.GL_FLOAT, g_cube.strides[0], ctypes.c_void_p(12))
    
    gl.glEnableClientState(gl.GL_NORMAL_ARRAY)
    gl.glNormalPointer(gl.GL_FLOAT, g_cube.strides[0], ctypes.c_void_p(20))
    
    gl.glClientActiveTexture(gl.GL_TEXTURE1)
    gl.glEnableClientState(gl.GL_TEXTURE_COORD_ARRAY)
    gl.glTexCoordPointer(4, gl.GL_FLOAT, g_cube.strides[0], ctypes.c_void_p(32))
    
    #gl.glDrawArrays(gl.GL_QUADS, 0, 24)

    a = (gl.GLfloat * 16)()
    mvm = gl.glGetFloatv(gl.GL_PROJECTION_MATRIX, a)
    #print (list(a))
    
    
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, g_vertexIndexBuffer)
    gl.glTranslate(0.5,0,0)
    a = (gl.GLfloat * 16)()
    mvm = gl.glGetFloatv(gl.GL_PROJECTION_MATRIX, a)
    #print (list(a))
    gl.glDrawElements(gl.GL_QUADS, 24, gl.GL_UNSIGNED_INT, None)
    
    gl.glTranslate(2,0,0)
    gl.glScale(0.5, 0.5, 0.5)
    gl.glDrawElements(gl.GL_QUADS, 24, gl.GL_UNSIGNED_INT, None)
    
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
    
    #glut.glutSolidSphere(0.3,20,20)
    
    gl.glDisableClientState(gl.GL_TEXTURE_COORD_ARRAY)
    gl.glDisableClientState(gl.GL_NORMAL_ARRAY)
    gl.glClientActiveTexture(gl.GL_TEXTURE0)
    gl.glDisableClientState(gl.GL_TEXTURE_COORD_ARRAY)
    gl.glDisableClientState(gl.GL_VERTEX_ARRAY)

    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    gl.glActiveTexture(gl.GL_TEXTURE1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    gl.glDisable(gl.GL_TEXTURE_2D)

    gl.glActiveTexture(gl.GL_TEXTURE0)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    gl.glDisable(gl.GL_TEXTURE_2D)    

def RenderFrame():
    global g_frames
    
    # Setup view port
    gl.glViewport(0, 0, g_windowWidth, g_windowHeight)
    gl.glClearColor(0.0, 0.0, 0.0, 0.0)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT) 
    
   
    # Setup perspective projection matrix
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    glu.gluPerspective(CAMERA_FOVY, g_windowWidth / g_windowHeight, CAMERA_ZNEAR, CAMERA_ZFAR)

    # Setup view matrix
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()
    glu.gluLookAt(CAMERA_RADIUS * sind(view_rotx), view_roty, CAMERA_RADIUS * cosd(view_rotx),  0, 0, 0,  0.0, 1.0, 0.0)
    
    gl_worldview_matrix = (gl.GLfloat * 16)()
    gl.glGetFloatv(gl.GL_MODELVIEW_MATRIX, gl_worldview_matrix)
    #print ("gl_worldview_matrix=",list(gl_worldview_matrix))    
    
    
    # Setup lighting and shaders
    gl.glPushAttrib(gl.GL_LIGHTING_BIT)
    

    gl.glUseProgram(g_disneyShaderProgram)
    gl.glUniformMatrix4fv(gl.glGetUniformLocation(g_disneyShaderProgram, "worldViewMatrix"), 1, gl.GL_FALSE, gl_worldview_matrix)
    
    gl.glUniform1i(gl.glGetUniformLocation(g_disneyShaderProgram, "light1Type"), light1Type)
    gl.glUniform3fv(gl.glGetUniformLocation(g_disneyShaderProgram, "light1Color"),1, [1.0, 1.0, 1.0])
    gl.glUniform3fv(gl.glGetUniformLocation(g_disneyShaderProgram, "light1WorldPointDir"),1, [0.0, 0, 2.1])
    gl.glUniform1f(gl.glGetUniformLocation(g_disneyShaderProgram, "light1Brightness"), 1)

    gl.glUniform1i(gl.glGetUniformLocation(g_disneyShaderProgram, "light2Type"), light2Type)
    gl.glUniform3fv(gl.glGetUniformLocation(g_disneyShaderProgram, "light2Color"),1, [0.5, 0.25, 0.25])
    gl.glUniform3fv(gl.glGetUniformLocation(g_disneyShaderProgram, "light2WorldPointDir"),1, [-2, 0, 0.0])
    gl.glUniform1f(gl.glGetUniformLocation(g_disneyShaderProgram, "light2Brightness"), 1.0)
    
    gl.glUniform1i(gl.glGetUniformLocation(g_disneyShaderProgram, "light3Type"), light3Type)
    gl.glUniform3fv(gl.glGetUniformLocation(g_disneyShaderProgram, "light3Color"),1, [0.3, 0.45, 0.3])
    gl.glUniform3fv(gl.glGetUniformLocation(g_disneyShaderProgram, "light3WorldPointDir"),1, [1, 1.1, 0.0])
    gl.glUniform1f(gl.glGetUniformLocation(g_disneyShaderProgram, "light3Brightness"), 1.0)

    gl.glUniform1i(gl.glGetUniformLocation(g_disneyShaderProgram, "light4Type"), light4Type)
    gl.glUniform3fv(gl.glGetUniformLocation(g_disneyShaderProgram, "light4Color"),1, [0.5, 0.5, 0.55])
    gl.glUniform3fv(gl.glGetUniformLocation(g_disneyShaderProgram, "light4WorldPointDir"),1, [0, -1.3, 0.0])
    gl.glUniform1f(gl.glGetUniformLocation(g_disneyShaderProgram, "light4Brightness"), 1.0)

    gl.glUniform1i(gl.glGetUniformLocation(g_disneyShaderProgram, "light5Type"), light5Type)
    gl.glUniform3fv(gl.glGetUniformLocation(g_disneyShaderProgram, "light5Color"),1, [0.6, 0.15, 0.6])
    gl.glUniform3fv(gl.glGetUniformLocation(g_disneyShaderProgram, "light5WorldPointDir"),1, [1, 1, 1.0])
    gl.glUniform1f(gl.glGetUniformLocation(g_disneyShaderProgram, "light5Brightness"), 1.0)
    
    gl.glUniform1i(gl.glGetUniformLocation(g_disneyShaderProgram, "colorMap"), 0)
    gl.glUniform1i(gl.glGetUniformLocation(g_disneyShaderProgram, "useColorTexture"), 1 )
    gl.glUniform1i(gl.glGetUniformLocation(g_disneyShaderProgram, "normalMap"), 1)
    gl.glUniform1i(gl.glGetUniformLocation(g_disneyShaderProgram, "useNormalTexture"), 1)
    gl.glUniform1i(gl.glGetUniformLocation(g_disneyShaderProgram, "roughnessMap"), 2)
    gl.glUniform1i(gl.glGetUniformLocation(g_disneyShaderProgram, "useRoughnessTexture"), 1)
    gl.glUniform1i(gl.glGetUniformLocation(g_disneyShaderProgram, "metalnessMap"), 3)
    gl.glUniform1i(gl.glGetUniformLocation(g_disneyShaderProgram, "useMetalnessTexture"), 1)        
    
    gl.glUniform3fv(gl.glGetUniformLocation(g_disneyShaderProgram, "materialBaseColor"),1, [.82, .67, .16])
    gl.glUniform1f(gl.glGetUniformLocation(g_disneyShaderProgram, "materialMetallic"), 0.2)
    gl.glUniform1f(gl.glGetUniformLocation(g_disneyShaderProgram, "materialSubsurface"), 0.0)
    gl.glUniform1f(gl.glGetUniformLocation(g_disneyShaderProgram, "materialSpecular"), 0.4)
    gl.glUniform1f(gl.glGetUniformLocation(g_disneyShaderProgram, "materialRoughness"), 0.0)
    gl.glUniform1f(gl.glGetUniformLocation(g_disneyShaderProgram, "materialSpecularTint"), 0.0)
    gl.glUniform1f(gl.glGetUniformLocation(g_disneyShaderProgram, "materialAnisotropic"), 0.0)
    gl.glUniform1f(gl.glGetUniformLocation(g_disneyShaderProgram, "materialSheen"), 0.0)
    gl.glUniform1f(gl.glGetUniformLocation(g_disneyShaderProgram, "materialSheenTint"), 0.5)
    gl.glUniform1f(gl.glGetUniformLocation(g_disneyShaderProgram, "materialClearcoat"), 0.0)
    gl.glUniform1f(gl.glGetUniformLocation(g_disneyShaderProgram, "materialClearcoatGloss"), 0.0)

       
    
    # Render the cube
    #roll
    gl.glRotatef(-model_rotz, 1, 0, 0)
    #yaw
    gl.glRotatef(-model_roty, 0, 1, 0)
    #pitch
    gl.glRotatef(model_rotx, 0, 0, 1)

    
    RenderCube()

    
    gl.glUseProgram(0)
    
    gl.glPopAttrib()
    
    glut.glutSwapBuffers()
    
    g_frames = g_frames + 1
    
    return


if __name__ == '__main__': 
    main()
