import moderngl
from PIL import Image
import numpy as np
import pytorch3d.io


class SingleObjectRender:
    def __init__(self, model_path=None, objs=None, frame_size=(512, 512),
                 enable_cull=True, enable_depth_test=True, enable_texture=True):
        # create OpenGL context, first try a available context, create one if found none
        try:
            self.ctx = moderngl.create_context(standalone=True, backend='egl')
        except:
            print('create egl context error.')

        assert model_path is not None or objs is not None

        self.enable_texture = enable_texture

        self.default_depth = 1.0

        # load model and texture file
        if model_path is None:
            self.add_objs(objs)
        else:
            self.load_geometry(model_path)

        # generate framebuffer object
        self.frame_size = frame_size
        self.gen_fbo(frame_size)

        # gl program
        self.gen_program()

        # texture and sampler
        self.gen_texture_sampler()

        # vertex array
        self.gen_vao()

        render_flag = moderngl.NOTHING
        if enable_cull:
            render_flag |= moderngl.CULL_FACE

        if enable_depth_test:
            render_flag |= moderngl.DEPTH_TEST

        self.ctx.enable(render_flag)

        # self.ctx.depth_func = '>'
        # self.ctx.cull_face = 'front'

        self.set_near_far()

    def set_near_mode(self):
        self.ctx.depth_func = '<='
        self.default_depth = 1.0

    def set_far_mode(self):
        self.ctx.depth_func = '>='
        self.default_depth = 0.0

    def add_objs(self, res):
        # calculate vertex normal
        # vn = igl.per_vertex_normals(v, f)
        v = res.verts_list()[0].numpy()
        f = res.faces_list()[0].numpy().astype(np.int32)
        vn = res.verts_normals_list()[0].numpy()

        if self.enable_texture:
            tc = res.textures.verts_uvs_list()[0].numpy()
            im_data = res.textures.maps_padded()[0].numpy() * 255.0
            im_data = im_data.astype(np.uint8)

            ftc = res.textures.faces_uvs_list()[0].numpy()
            im_data = np.flipud(im_data)

        self.geometry = {}

        # if texture coord available, merge vertex&texture coord info and ignore f info

        v = v[f.reshape(-1)]
        vn = vn[f.reshape(-1)]
        if self.enable_texture:
            tc = tc[ftc.reshape(-1)]

        f = None

        self.geometry['v'] = v
        self.geometry['normal'] = vn
        if self.enable_texture:
            self.geometry['texture_coord'] = tc
            self.geometry['texture_data'] = im_data

    def load_geometry(self, model_path):
        res = pytorch3d.io.load_objs_as_meshes([model_path], load_textures=self.enable_texture)
        self.add_objs(res)

    def gen_fbo(self, fbo_size):
        '''
        generate a default framebuffer and attach three buffer in order,
        
        add self.fbo member
        add self.fbo_attachmen_info, may help read out buffer data later
            in which the key format is [semantic info] : [attachment point]
        '''

        color_tex = self.ctx.texture(fbo_size, components=4, dtype='f4')
        normal_tex = self.ctx.texture(fbo_size, components=3, dtype='f4')
        depth_tex = self.ctx.texture(fbo_size, components=1, dtype='f4')

        self.fbo_attachment_info = {'color:0': color_tex,
                                    'normal_in_view:1': normal_tex,
                                    'z_depth:2': depth_tex}

        self.fbo = self.ctx.framebuffer(color_attachments=(color_tex, normal_tex, depth_tex),
                                        depth_attachment=self.ctx.depth_renderbuffer(fbo_size))
        self.fbo.use()

    def gen_program(self):
        '''
        generate a gl program for render
        
        add a self.prog member
        '''

        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 mvp;
                uniform mat3 opencv_world2cam_rot;

                in vec2 in_tc;
                in vec3 in_vert;
                in vec3 in_normal;

                out vec3 v_normal;
                out float v_depth;
                out vec2 v_tex_coord;

                void main() {
                    gl_Position = mvp * vec4(in_vert, 1.0);
                    v_normal = opencv_world2cam_rot * in_normal;
                    v_depth = gl_Position[2];
                    v_tex_coord = in_tc;
                }
            ''',
            fragment_shader='''
                #version 330
                uniform sampler2D tex0;

                in vec2 v_tex_coord;
                in vec3 v_normal;
                in float v_depth;

                layout(location=0) out vec4 p_color;
                layout(location=1) out vec3 p_normal;
                layout(location=2) out float p_depth;
                void main() {
                    p_normal = v_normal;
                    p_depth = v_depth;
                    p_color = vec4(texture(tex0, v_tex_coord).rgb, 1.0);
                }
            ''')

        # set value of sampler in glsl shader 
        self.prog['tex0'] = 0

    def gen_texture_sampler(self):
        '''
        if geometry has texture infomation, generate texture and sampler
        
        (maybe) add self.sampler member
        (maybe) add self.tex2d member
        '''

        if hasattr(self, 'geometry'):
            if 'texture_data' in self.geometry:
                h, w, c = self.geometry['texture_data'].shape[0:3]
                self.tex2d = self.ctx.texture((w, h), c,
                                              self.geometry['texture_data'].tobytes(),
                                              dtype='f1')
                self.tex2d.build_mipmaps()

                self.sampler = self.ctx.sampler(filter=(moderngl.LINEAR_MIPMAP_LINEAR,
                                                        moderngl.NEAREST),
                                                texture=self.tex2d)

                self.sampler.use(location=0)  # consistent with prog['tex0']

    def gen_vao(self):
        '''
        generate vertex array to define (meta) attribute data 
        generate vertex buffer to save actually attribute data 
        
        add self.vao
        add self.vao_content
        '''
        if not hasattr(self, 'geometry'):
            raise Exception('no geometry yet')

        vbo_v = self.ctx.buffer(self.geometry['v'].tobytes())
        self.vao_content = [(vbo_v, '3f4', 'in_vert')]

        if 'normal' in self.geometry:
            vbo_normal = self.ctx.buffer(self.geometry['normal'].tobytes())
            self.vao_content.append((vbo_normal, '3f4', 'in_normal'))

        if 'texture_coord' in self.geometry:
            vbo_tc = self.ctx.buffer(self.geometry['texture_coord'].tobytes())
            self.vao_content.append((vbo_tc, '2f4', 'in_tc'))

        # index buffer if face info still available
        if 'f' in self.geometry:
            ibo_f = self.ctx.buffer(self.geometry['f'])
            self.vao = self.ctx.vertex_array(self.prog, self.vao_content, ibo_f)
        else:
            self.vao = self.ctx.vertex_array(self.prog, self.vao_content)

    # the following API maybe called several times
    def set_near_far(self, near=0.01, far=100):
        self.near = near
        self.far = far

    def set_MVP(self, opencv_K, cam2world_R, cam2world_t, view_size=None):
        if view_size is None:
            view_size = self.frame_size

        mv = np.zeros((4, 4), dtype=np.float32)
        mv[0:3, 0:3] = cam2world_R.T
        mv[0:3, 3] = -cam2world_R.T.dot(cam2world_t)
        mv[3, 3] = 1

        cv2gl = np.zeros((4, 4), dtype=np.float32)
        cv2gl[3, 3] = 1
        cv2gl[0, 0] = 1
        cv2gl[1, 1] = -1
        cv2gl[2, 2] = -1

        # calculate left/right/top/bottom in gl frame, 
        # namely the top and bottom is the minus of that in opencv frame
        left = -(opencv_K[0, 2] * self.near) / opencv_K[0, 0]
        right = (view_size[0] - opencv_K[0, 2]) * self.near / opencv_K[0, 0]
        top = opencv_K[1, 2] * self.near / opencv_K[1, 1]
        bottom = -(view_size[1] - opencv_K[1, 2]) * self.near / opencv_K[1, 1]

        p = np.zeros((4, 4), dtype=np.float32)
        p[0, 0] = 2 * self.near / (right - left)
        p[0, 2] = (right + left) / (right - left)
        p[1, 1] = 2 * self.near / (top - bottom)
        p[1, 2] = (top + bottom) / (top - bottom)
        p[2, 2] = -(self.near + self.far) / (self.far - self.near)
        p[2, 3] = -2 * self.far * self.near / (self.far - self.near)
        p[3, 2] = -1

        mvp = p.dot(cv2gl).dot(mv)

        # set the 'mvp' uniform in shader, note that shader mat is COL-MAJOR
        self.prog['mvp'].write(mvp.T.tobytes())

        # also set the 'opencv_world2cam_rot' uniform in shader, this will be used to rotate normal
        world2cam_rot = cam2world_R.T.astype(np.float32)
        self.prog['opencv_world2cam_rot'].write(world2cam_rot.T.tobytes())

    def render(self):
        self.ctx.clear(0.0, 0.0, 0.0, 0.0, depth=self.default_depth)
        self.vao.render()

    def read(self):
        '''
        read all three attachment in framebuffer
        
        each image maybe undefined value if render condition violated
        E.X. if loading a model without texture, the color-slot image is unexpected
        '''

        ret_frame = {}
        for name, buffer in self.fbo_attachment_info.items():
            name, attach_point = map(lambda x: x.strip(), name.split(':'))
            attach_point = int(attach_point)

            frame_data = np.frombuffer(self.fbo.read(components=buffer.components,
                                                     dtype=buffer.dtype, attachment=attach_point),
                                       dtype=buffer.dtype)
            frame_data = frame_data.reshape(self.frame_size[1], self.frame_size[0], buffer.components)
            frame_data = np.flipud(frame_data)

            # compress color
            if name == 'color':
                frame_data = (frame_data * 255).astype(np.uint8)

            ret_frame[name] = frame_data.squeeze()

        return ret_frame

    def release(self):
        '''
        seems that the resource need to be released manually
        '''
        # release framebuffer and attachments
        for tex in self.fbo.color_attachments:
            tex.release()

        self.fbo.depth_attachment.release()
        self.fbo.release()

        # release program 
        self.prog.release()

        # release texture and sampler
        if 'texture_data' in self.geometry:
            self.tex2d.release()
            self.sampler.release()

        # release vertex array related
        if self.vao.index_buffer:
            self.vao.index_buffer.release()

        for i in self.vao_content:
            i[0].release()

        self.vao.release()

        # try to release context
        self.ctx.release()


def render_near_far_depth(render, opencv_K, cam2world_R, cam2world_t, view_size=None):
    render.set_MVP(opencv_K, cam2world_R, cam2world_t, view_size)
    render.set_near_mode()
    render.render()
    res = render.read()
    near_depth = res['z_depth']

    render.set_far_mode()
    render.render()
    res = render.read()
    far_depth = res['z_depth']

    return near_depth.copy(), far_depth.copy()
