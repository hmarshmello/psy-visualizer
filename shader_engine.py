import numpy as np
import moderngl


class ShaderRenderer:
    def __init__(self, width, height, vert_path, frag_path):
        self.width = width
        self.height = height
        self.ctx = self._create_context()
        self.program = self._load_program(vert_path, frag_path)
        self.vbo = self.ctx.buffer(
            np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype='f4').tobytes()
        )
        self.vao = self.ctx.simple_vertex_array(self.program, self.vbo, 'in_position')
        self.color_texture = self.ctx.texture((width, height), 3)
        self.fbo = self.ctx.framebuffer(color_attachments=[self.color_texture])

    def _create_context(self):
        errors = []
        for backend in (None, 'egl'):
            try:
                if backend:
                    return moderngl.create_standalone_context(backend=backend, require=330)
                return moderngl.create_standalone_context(require=330)
            except Exception as e:
                errors.append(f'{backend}: {e}')
        raise RuntimeError('No headless OpenGL context available: ' + ' | '.join(errors))

    def _load_program(self, vert_path, frag_path):
        with open(vert_path) as f:
            vert_src = f.read()
        with open(frag_path) as f:
            frag_src = f.read()
        return self.ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)

    def render_frame(self, uniforms):
        self.fbo.use()
        self.fbo.clear(0.0, 0.0, 0.0, 1.0)
        self.program['u_resolution'].value = (float(self.width), float(self.height))
        for key, val in uniforms.items():
            if key in self.program:
                self.program[key].value = val
        self.vao.render(mode=moderngl.TRIANGLE_STRIP)
        raw = self.fbo.read(components=3, alignment=1)
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 3)
        frame = np.flipud(frame)
        return np.ascontiguousarray(frame).tobytes()

    def release(self):
        self.fbo.release()
        self.color_texture.release()
        self.vao.release()
        self.vbo.release()
        self.program.release()
        self.ctx.release()
