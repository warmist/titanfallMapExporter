# McSimps Titanfall Map Exporter Tool
# Website: https://will.io/
# Modded(butchered) to output ply by Warmist
#   disclaimer: i don't know python...

import struct
from enum import Enum
import os
import math

#settings
map_name = 'mp_wargames'
map_unpacked_path='G:\\tmp\\mp_wargames\\' #path to unpacked bsp folder with "models" and "maps" subfolder
common_model_path_prefix='G:\\tmp\\mp_common\\' # path to unpacked englishclient_mp_common.bsp.pak000_dir.vpk (only models folder is needed)
#not settings
model_path_prefix=map_unpacked_path
map_path_prefix=map_unpacked_path+'\\maps\\' # path to unpacked bsp
dump_base = map_path_prefix+"out\\"
map_path =  map_path_prefix + map_name + '.bsp'

#mprt has bad angles, it might be wrongly rotated, but left it for future generations to admire...
mprt_path=""

def read_null_string(f):
    chars = []
    while True:
        c = f.read(1).decode('ascii')
        if c == chr(0) or c=='':
            return ''.join(chars)
        chars.append(c)

class LumpElement:
    @staticmethod
    def get_size():
        raise NotImplementedError()

class TextureData(LumpElement):
    def __init__(self, data):
        self.string_table_index = struct.unpack_from('<I', data, 12)[0]

    @staticmethod
    def get_size():
        return 36

class BumpLitVertex(LumpElement):
    def __init__(self, data):
        self.vertex_pos_index = struct.unpack_from('<I', data, 0)[0]
        self.vertex_normal_index = struct.unpack_from('<I', data, 4)[0]
        self.texcoord0 = struct.unpack_from('<ff', data, 8) # coord into albedo, normal, gloss, spec
        self.texcoord5 = struct.unpack_from('<ff', data, 20) # coord into lightmap

    @staticmethod
    def get_size():
        return 44

class UnlitVertex(LumpElement):
    def __init__(self, data):
        self.vertex_pos_index = struct.unpack_from('<I', data, 0)[0]
        self.vertex_normal_index = struct.unpack_from('<I', data, 4)[0]
        self.texcoord0 = struct.unpack_from('<ff', data, 8)

    @staticmethod
    def get_size():
        return 20

class UnlitTSVertex(LumpElement):
    def __init__(self, data):
        self.vertex_pos_index = struct.unpack_from('<I', data, 0)[0]
        self.vertex_normal_index = struct.unpack_from('<I', data, 4)[0]
        self.texcoord0 = struct.unpack_from('<ff', data, 8)

    @staticmethod
    def get_size():
        return 28

class MaterialSortElement(LumpElement):
    def __init__(self, data):
        self.texture_index = struct.unpack_from('<H', data, 0)[0]
        self.vertex_start_index = struct.unpack_from('<I', data, 8)[0]

    @staticmethod
    def get_size():
        return 12

class VertexType(Enum):
    LIT_FLAT = 0
    UNLIT = 1
    LIT_BUMP = 2
    UNLIT_TS = 3

class MeshElement(LumpElement):
    def __init__(self, data):
        self.indices_start_index = struct.unpack_from('<I', data, 0)[0]
        self.num_triangles = struct.unpack_from('<H', data, 4)[0]
        self.material_sort_index = struct.unpack_from('<H', data, 22)[0]
        self.flags = struct.unpack_from('<I', data, 24)[0]

    def get_vertex_type(self):
        temp = 0
        if self.flags & 0x400:
            temp |= 1
        if self.flags & 0x200:
            temp |= 2
        return VertexType(temp)

    @staticmethod
    def get_size():
        return 28

class PhyHeader(LumpElement): # Header in SourceIO
    def __init__(self, data):
        self.size = struct.unpack_from('<I', data, 0)[0]
        self.id = struct.unpack_from('<I', data, 4)[0]
        self.solidCount = struct.unpack_from('<I', data, 8)[0]
        self.checkSum = struct.unpack_from('<I', data, 12)[0]

    @staticmethod
    def get_size():
        return 16

class CompactSurfHeader(LumpElement):
    def __init__(self, data):
        self.size = struct.unpack_from('<I', data, 0)[0]
        self.id = struct.unpack_from('<I', data, 4)[0]
        self.version = struct.unpack_from('<H', data, 8)[0]
        self.modelType = struct.unpack_from('<H', data, 10)[0]
        self.surfaceSize=struct.unpack_from('<I', data, 12)[0]
        self.dragAxisAreas=struct.unpack_from('<3f', data, 16)
        self.axisMapSize=struct.unpack_from('<I', data, 28)[0]
        #collision model here
        self.cm_values=struct.unpack_from('<7f', data, 32)
        self.surface,self.offset_tree,*self.pad=struct.unpack_from('<4I',data,60)
        #some unk stuff here
        self.IPVS=struct.unpack_from('<I', data, 76)[0]

    @staticmethod
    def get_size():
        return 80
    @staticmethod
    def get_model_start():
        return 28

class PhyFaceHeader(LumpElement): #SourceIO ConvexLeaf
    def __init__(self, data):
        self.size = struct.unpack_from('<I', data, 0)[0]
        self.boneidx = struct.unpack_from('<I', data, 4)[0]
        self.unk=struct.unpack_from('<I', data, 8)[0]
        self.tri_count=struct.unpack_from('<I', data, 12)[0]

    @staticmethod
    def get_size():
        return 16

class TreeNode():
    def __init__(self, data):
        self.right_node_offset, self.convex_offset, *self.floats = struct.unpack_from('2i5f',data,0)

        self.left_node: Optional[TreeNode] = None
        self.right_node: Optional[TreeNode] = None
        self.convex_leaf: Optional[ConvexLeaf] = None

    @staticmethod
    def get_size():
        return 24

def load_tree(data,root_offset):
    pass
class MDL_Mesh():
    def __init__(self):
        self.tris=[]
        self.verts=[]
    def get_verts_transformed(self,origin,angles,scale):
        ca=math.cos(angles[0])
        cb=math.cos(angles[1])
        cc=math.cos(angles[2])

        sa=math.sin(angles[0])
        sb=math.sin(angles[1])
        sc=math.sin(angles[2])

        Axx = ca*cb
        Axy = ca*sb*sc - sa*cc
        Axz = ca*sb*cc + sa*sc

        Ayx = sa*cb;
        Ayy = sa*sb*sc + ca*cc;
        Ayz = sa*sb*cc - ca*sc;

        Azx = -sb;
        Azy = cb*sc;
        Azz = cb*cc;
        verts=[]
        for v in self.verts:
            nv=[0,0,0]
            nv[0]=(Axx*v[0]+Axy*v[1]+Axz*v[2])*scale+origin[0]
            nv[1]=(Ayx*v[0]+Ayy*v[1]+Ayz*v[2])*scale+origin[1]
            nv[2]=-(Azx*v[0]+Azy*v[1]+Azz*v[2])*scale+origin[2]
            verts.append(nv)
        return verts
    def get_tris_offset(self,offset):
        ret=[]
        for t in self.tris:
            ret.append([t[0]+offset,t[2]+offset,t[1]+offset])
        return ret

def dump_ply(verts,tris,fname):
    with open(fname, 'wb') as f:
            f.write(b"ply\nformat binary_little_endian 1.0\n");
            v_str="element vertex {}\n".format(len(verts))
            f.write(bytes(v_str,encoding='utf8'))
            f.write(b"property float x\nproperty float y\nproperty float z\n")
            f_str="element face {}\n".format(len(tris))
            f.write(bytes(f_str,encoding='utf8'))
            f.write(b"property list uchar int vertex_index\nend_header\n")
            for v in verts:
                f.write(struct.pack("<fff",v[0],v[1],v[2]))

            for t in tris:
                f.write(struct.pack("<cIII",b'\x03',t[0], t[1], t[2]))


mdl_cache={}
def load_mdl_file(path):
    ret_mesh=MDL_Mesh()
    with open(path,"rb") as f:
        print("Loading:",path)
        f.seek(0x6c)
        bbox_min=struct.unpack_from('<fff', f.read(12),0)
        bbox_max=struct.unpack_from('<fff', f.read(12),0)
        #print(bbox_max,bbox_min)
        #print((bbox_max[0]-bbox_min[0],bbox_max[1]-bbox_min[1],bbox_max[2]-bbox_min[2]))
        f.seek(0x1b8) # seek to phyOffset
        phy_offset=struct.unpack_from('<I', f.read(4), 0)[0]
        if phy_offset==0: #no phy, don't add the file
            return
        #print(path,phy_offset)
        f.seek(phy_offset)
        head0=PhyHeader(f.read(PhyHeader.get_size()))
        #print("Solid count:",head0.solidCount)
        next_header=phy_offset+PhyHeader.get_size()
        vertex_pos=next_header+head0.size
        max_vert=0
        bbox_phy_min=[99999,99999,99999]
        bbox_phy_max=[-99999,-99999,-99999]
        for i in range(head0.solidCount):
            f.seek(next_header)
            chead=CompactSurfHeader(f.read(CompactSurfHeader.get_size()))
            next_header+=chead.size
            cur_pos=phy_offset+PhyHeader.get_size()
            while cur_pos<vertex_pos:
                ph=PhyFaceHeader(f.read(PhyFaceHeader.get_size()))
                #print(hex(ph.unk))
                vertex_pos=cur_pos+ph.size
                cur_pos+=PhyFaceHeader.get_size()
                for j in range(ph.tri_count): #ConvexTriangle in SourceIO
                    f.seek(4,1) # u8 tri_index,u8 unk,u16 unk
                    cur_pos+=4
                    tri=[]
                    for k in range(3):
                        vert_id=struct.unpack_from('<H', f.read(2), 0)[0]
                        if vert_id>max_vert:
                            max_vert=vert_id
                        tri.append(vert_id)
                        f.seek(2,1) #unk per tri
                        cur_pos+=4
                    #print(tri)
                    if not(ph.unk&1): #skip top level convex hull
                        ret_mesh.tris.append(tri)
            for j in range(max_vert+1):
            #while cur_pos<next_header:
                v=[]
                for k in range(3):
                    vf=struct.unpack_from('<f', f.read(4), 0)[0]
                    v.append(vf)
                    if bbox_phy_max[k]<vf:
                        bbox_phy_max[k]=vf
                    if bbox_phy_min[k]>vf:
                        bbox_phy_min[k]=vf
                f.seek(4,1)
                cur_pos+=16
                ret_mesh.verts.append([v[0],v[2],v[1]]) # exchange y and z, because bbox'es don't match the mdl!
            #print(max_vert)
        #print(len(ret_mesh.tris),len(ret_mesh.verts))
        #print(((bbox_phy_max[0]-bbox_phy_min[0])*40,(bbox_phy_max[1]-bbox_phy_min[1])*40,(bbox_phy_max[2]-bbox_phy_min[2])*40))
        quit()
        #dump_ply(ret_mesh.verts,ret_mesh.tris,"out.ply")
        #print(ret_mesh)
        return ret_mesh

def load_or_get_mdl(path):
    if path in mdl_cache:
        return mdl_cache[path]
    fullpath=model_path_prefix+path
    mexist=os.path.exists(fullpath)
    if not mexist:
        fullpath=common_model_path_prefix+path
    mexist=os.path.exists(fullpath)
    if not mexist:
        print("File not found:",path)
        quit()
    #print("Loading mdl:",fullpath)
    #print("Exists:",mexist)
    mdl_cache[path]=load_mdl_file(fullpath)
    return mdl_cache[path]

mdl_verts=[]
mdl_tris=[]
load_props=True

def load_mprt_file():
    print("Loading mprt file...")
    with open(mprt_path+map_name+".mprt","rb") as f:
        f.seek(12)
        while True:
            #mdl_path=''.join(iter(lambda: f.read(1).decode('ascii'), '\x00'))
            mdl_path=read_null_string(f)
            if not mdl_path:
                break
            mdl=load_or_get_mdl(mdl_path)
            origin=struct.unpack_from('<fff',f.read(12),0)
            angles=struct.unpack_from('<fff',f.read(12),0)
            #print(angles)
            angles=(math.pi*((angles[0])/180),math.pi*((angles[2])/180),math.pi*((angles[1])/180))
            scale=struct.unpack_from('<f',f.read(4),0)[0]
            #print(origin,angles,scale)
            if not mdl: #no physics: skip
                continue
            #print("Loading:",mdl_path,scale)
            offset=len(mdl_verts)
            mdl_verts.extend(mdl.get_verts_transformed(origin,angles,scale*40))
            mdl_tris.extend(mdl.get_tris_offset(offset))

def load_prop_lump():
    print("Loading prop lump...")
    with open(map_path+".0023.bsp_lump",'rb') as f:
        f.seek(0x14)
        name_count=struct.unpack_from('<I',f.read(4),0)[0]
        model_names = struct.iter_unpack("128s", f.read(128 * name_count))
        model_names = [t[0].replace(b"\0", b"").decode() for t in model_names]
        print("loaded {} prop names".format(name_count))
        prop_count=struct.unpack_from('<I',f.read(4),0)[0]
        f.seek(8,1)
        for i in range(prop_count):
            origin=struct.unpack_from('<fff',f.read(12),0)
            angles=struct.unpack_from('<fff',f.read(12),0)
            # 0 1 2 crane off axis!
            # 0 2 1 crane off axis!
            # 2 0 1 crane off axis!
            # 2 1 0 crane off axis!

            # 1 0 2

            # 1 2 0
            angles=(math.pi*((angles[1])/180),math.pi*((angles[2])/180),math.pi*((angles[0])/180))
            scale=struct.unpack_from('<f',f.read(4),0)[0]
            mdl_name_idx=struct.unpack_from('<H',f.read(2),0)[0]
            f.seek(34,1) # skip rest of struct
            #print(mdl_name_idx)
            #print("model_names[mdl_name_idx]:",model_names[mdl_name_idx])
            mdl=load_or_get_mdl(model_names[mdl_name_idx])
            if not mdl:
                continue
            offset=len(mdl_verts)
            mdl_verts.extend(mdl.get_verts_transformed(origin,angles,scale*40))
            mdl_tris.extend(mdl.get_tris_offset(offset))

if load_props:
    if mprt_path=="":
        load_prop_lump()
    else:
        load_mprt_file()
    #dump_ply(mdl_verts,mdl_tris,"out1.ply")
    print("Final mdl verts, tris:",len(mdl_verts),len(mdl_tris))

# Read all vertex position data
print("Reading vertex position data...")
with open(map_path + '.0003.bsp_lump', 'rb') as f:
    data = f.read()
    vertex_positions = [struct.unpack_from('<fff', data, i * 12) for i in range(len(data) // 12)]

# Read all vertex normals
print("Reading vertex normal data...")
with open(map_path + '.001e.bsp_lump', 'rb') as f:
    data = f.read()
    vertex_normals = [struct.unpack_from('<fff', data, i * 12) for i in range(len(data) // 12)]

# Read indices
print("Reading indices...")
with open(map_path + '.004f.bsp_lump', 'rb') as f:
    data = f.read()
    indices = [struct.unpack_from('<H', data, i * 2)[0] for i in range(len(data) // 2)]

# Read texture information
print("Reading texture information...")
with open(map_path + '.002c.bsp_lump', 'rb') as f:
    data = f.read()
    texture_string_offets = [struct.unpack_from('<I', data, i * 4)[0] for i in range(len(data) // 4)]

with open(map_path + '.002b.bsp_lump', 'rb') as f:
    texture_strings = []
    for offset in texture_string_offets:
        f.seek(offset)
        texture_strings.append(read_null_string(f))

textures = []
with open(map_path + '.0002.bsp_lump', 'rb') as f:
    data = f.read()
    elem_size = TextureData.get_size()
    for i in range(len(data) // elem_size):
        textures.append(TextureData(data[i*elem_size:(i+1)*elem_size]))

# Read bump lit vertices
print("Reading bump lit vertices...")
bump_lit_vertices = []
with open(map_path + '.0049.bsp_lump', 'rb') as f:
    data = f.read()
    elem_size = BumpLitVertex.get_size()
    for i in range(len(data) // elem_size):
        bump_lit_vertices.append(BumpLitVertex(data[i*elem_size:(i+1)*elem_size]))


# Read unlit vertices
print("Reading unlit vertices...")
unlit_vertices = []
unlit_vertex_bsp_lump = map_path + ".0047.bsp_lump"
if os.path.exists(unlit_vertex_bsp_lump):
    with open(unlit_vertex_bsp_lump, 'rb') as f:
        data = f.read()
        elem_size = UnlitVertex.get_size()
        for i in range(len(data) // elem_size):
            unlit_vertices.append(UnlitVertex(data[i*elem_size:(i+1)*elem_size]))

# Read unlit TS vertices
print("Reading unlit TS vertices...")
unlit_ts_vertices = []
unlit_ts_vertex_bsp_lump = map_path + ".004a.bsp_lump"
if os.path.exists(unlit_ts_vertex_bsp_lump ):
    with open(unlit_ts_vertex_bsp_lump , 'rb') as f:
      data = f.read()
      elem_size = UnlitTSVertex.get_size()
      for i in range(len(data) // elem_size):
          unlit_ts_vertices.append(UnlitTSVertex(data[i*elem_size:(i+1)*elem_size]))

vertex_arrays = [
    [],
    unlit_vertices,
    bump_lit_vertices,
    unlit_ts_vertices
]

# Read the material sort data
print("Reading material sort data...")
material_sorts = []
with open(map_path + '.0052.bsp_lump', 'rb') as f:
    data = f.read()
    elem_size = MaterialSortElement.get_size()
    for i in range(len(data) // elem_size):
        material_sorts.append(MaterialSortElement(data[i*elem_size:(i+1)*elem_size]))

# Read mesh information
print("Reading mesh data...")
meshes = []
with open(map_path + '.0050.bsp_lump', 'rb') as f:
    data = f.read()
    elem_size = MeshElement.get_size()
    for i in range(len(data) // elem_size):
        meshes.append(MeshElement(data[i*elem_size:(i+1)*elem_size]))

# Build combined model data
print("Building combined model data...")
combined_uvs = []
mesh_faces = []
texture_set = set()
for mesh_index in range(len(meshes)):
    faces = []
    mesh = meshes[mesh_index]
    if mesh.get_vertex_type().value==1 or mesh.get_vertex_type().value==3:
        continue
    mat = material_sorts[mesh.material_sort_index]
    texture_set.add(texture_strings[textures[mat.texture_index].string_table_index])
    for i in range(mesh.num_triangles * 3):
        vertex = vertex_arrays[mesh.get_vertex_type().value][mat.vertex_start_index + indices[mesh.indices_start_index + i]]
        combined_uvs.append(vertex.texcoord0)
        uv_idx = len(combined_uvs) - 1
        #faces.append((vertex.vertex_pos_index + 1, uv_idx + 1, vertex.vertex_normal_index + 1)) # obj files use 1 as index start
        faces.append((vertex.vertex_pos_index , uv_idx , vertex.vertex_normal_index )) #ply files use 0 as index start
    mesh_faces.append(faces)

# Build material files
# print('Building material files...')
# for i in range(len(textures)):
#     texture_string = texture_strings[textures[i].string_table_index]

#     # Work out the path to the actual texture
#     if os.path.isfile(dump_base + texture_string + '.png'):
#         path = dump_base + texture_string + '.png'
#     elif os.path.isfile(dump_base + texture_string + '_col.png'):
#         path = dump_base + texture_string + '_col.png'
#     else:
#         print('[!] Failed to find texture file for {}'.format(texture_string))
#         path = 'error.png'

    # # Write the material file
    # with open('{}\\tex{}.mtl'.format(map_name, i), 'w') as f:
    #      f.write('newmtl tex{}\n'.format(i))
    #      f.write('illum 1\n')
    #      f.write('Ka 1.0000 1.0000 1.0000\n')
    #      f.write('Kd 1.0000 1.0000 1.0000\n')
    #      f.write('map_Ka {}\n'.format(path))
    #      f.write('map_Kd {}\n'.format(path))


# Create obj file
print("Writing output file...")

# with open(map_name+".obj", 'w') as f:
#     f.write('o {}\n'.format(map_name))
#     for i in range(len(textures)):
#         f.write('mtllib tex{}.mtl\n'.format(i))

#     for v in vertex_positions:
#         f.write('v {} {} {}\n'.format(*v))

#     for v in vertex_normals:
#         f.write('vn {} {} {}\n'.format(*v))

#     for v in combined_uvs:
#         f.write('vt {} {}\n'.format(*v))

#     for i in range(len(mesh_faces)):
#         f.write('g {}\n'.format(i))
#         f.write('usemtl tex{}\n'.format(material_sorts[meshes[i].material_sort_index].texture_index))
#         faces = mesh_faces[i]
#         for i in range(len(faces) // 3):
#             f.write('f {}/{}/{} {}/{}/{} {}/{}/{}\n'.format(*faces[i*3], *faces[(i*3) + 1], *faces[(i*3) + 2]))

with open(map_name+".ply", 'wb') as f:
    f.write(b"ply\nformat binary_little_endian 1.0\n");
    v_str="element vertex {}\n".format(len(vertex_positions)+len(mdl_verts))
    #v_str="element vertex {}\n".format(len(mdl_verts))
    f.write(bytes(v_str,encoding='utf8'))
    f.write(b"property float x\nproperty float y\nproperty float z\n")

    num_triangles=0
    for mf in range(len(mesh_faces)):
        num_triangles+=len(mesh_faces[mf])//3

    f_str="element face {}\n".format(num_triangles+len(mdl_tris))
    #f_str="element face {}\n".format(len(mdl_tris))
    f.write(bytes(f_str,encoding='utf8'))
    f.write(b"property list uchar int vertex_index\nend_header\n")
    for v in vertex_positions:
        f.write(struct.pack("<fff",*v))

    for v in mdl_verts:
        f.write(struct.pack("<fff",*v))

    for i in range(len(mesh_faces)):
        faces = mesh_faces[i]
        for i in range(len(faces) // 3):
            f.write(struct.pack("<cIII",b'\x03',faces[i*3][0], faces[(i*3) + 1][0], faces[(i*3) + 2][0]))
    offset=len(vertex_positions)
    for t in mdl_tris:
        f.write(struct.pack("<cIII",b'\x03',t[0]+offset, t[1]+offset, t[2]+offset))