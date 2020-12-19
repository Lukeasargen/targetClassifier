import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageFont


class TargetGenerator():


    def __init__(self):
        self.shape_options = ["circle", "quartercircle", "semicircle", "square"]

        self.color_options = ['red', 'orange', 'yellow','green', 'blue',
                        'purple', 'brown', 'gray', 'white', 'black']

        self.num_classes = [
            len(self.shape_options),
            len(self.color_options)
        ]

    def color_to_hsv(self, color):
        options = {
            'red': lambda: (np.random.randint(0, 4), np.random.randint(50, 100), np.random.randint(40, 60)),
            'orange': lambda: (np.random.randint(9, 33), np.random.randint(50, 100), np.random.randint(40, 60)),
            'yellow': lambda: (np.random.randint(43, 55), np.random.randint(50, 100), np.random.randint(40, 60)),
            'green': lambda: (np.random.randint(75, 120), np.random.randint(50, 100), np.random.randint(40, 60)),
            'blue': lambda: (np.random.randint(200, 233), np.random.randint(50, 100), np.random.randint(40, 60)),
            'purple': lambda: (np.random.randint(266, 291), np.random.randint(50, 100), np.random.randint(40, 60)),
            'brown': lambda: (np.random.randint(13, 20), np.random.randint(25, 50), np.random.randint(22, 40)),
            'black': lambda: (np.random.randint(0, 360), np.random.randint(0, 12), np.random.randint(0, 6)),
            'gray': lambda: (np.random.randint(0, 360), np.random.randint(0, 12), np.random.randint(25, 60)),
            'white': lambda: (np.random.randint(0, 360), np.random.randint(0, 12), np.random.randint(80, 100))
        }
        h, s, l = options[color]()
        color_code = 'hsl(%d, %d%%, %d%%)' % (h, s, l)
        return color_code

    # TODO : add rotation
    def draw_shape(self, draw, img_size, target_size, shape, color):
        if shape == "circle":
            b = (img_size - target_size) // 2
            top = (b, b)
            bot = (img_size-b, img_size-b)
            draw.pieslice([top, bot], 0, 360, fill=color)
        elif shape == "quartercircle":
            b = (img_size - target_size) // 2
            top = (b-target_size, b-target_size)
            bot = (img_size-b, img_size-b)
            draw.pieslice([top, bot], 0, 90, fill=color)
        elif shape == "semicircle":
            b = (img_size - target_size) // 2
            h = target_size // 4
            top = (b, b-h)
            bot = (img_size-b, img_size-b-h)
            draw.pieslice([top, bot], 0, 180, fill=color)
        elif shape == "square":
            b = (img_size - target_size) // 2
            top = (b, b)
            bot = (img_size-b, img_size-b)
            draw.rectangle([top, bot], fill=color)

    def draw_target(self, img_size, target_size):
        img = Image.new('RGBA', size=(img_size, img_size), color=(0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Shape has to fit inside square of target_size
        shape_idx = np.random.randint(0, len(self.shape_options))
        # print("shape_idx :", shape_idx, self.shape_options[shape_idx])

        shape_color_idx = np.random.randint(0, len(self.color_options))
        # print("shape_color_idx :", shape_color_idx, self.color_options[shape_color_idx])
        shape_color = self.color_to_hsv(self.color_options[shape_color_idx])

        self.draw_shape(draw, img_size, target_size, self.shape_options[shape_idx], shape_color)

        img = img.convert('RGB')
        # img = np.array(img, dtype='uint8')

        label = {
            "shape": shape_idx,
            "color": shape_color_idx
        }
        return img, label


if __name__ == "__main__":

    gen = TargetGenerator()

    img_size = 32
    target_size = 20

    # x, y = gen.draw_target(img_size, target_size)
    # x.show()
    # exit()

    nimg = 10000
    mean = 0.0
    var = 0.0
    for i in range(nimg):
        # img in shape [W, H, C]
        img, y = gen.draw_target(img_size, target_size)
        # img = np.random.normal(size=(img_size, img_size, 3)) * 255
        # [1, C, H, W], expand so that the mean function can run on dim=0
        img = np.expand_dims((np.array(img) / 255.).transpose((2, 1, 0)), axis=0)
        mean += np.mean(img, axis=(0, 2, 3))
        var += np.var(img, axis=(0, 2, 3))  # you can add var, not std
    mean = mean/nimg
    std = np.sqrt(var/nimg)
    print("mean :", mean)
    print("std :", std)


    nrows = 32
    ncols = 64
    rows = []
    for i in range(nrows):
        row = []
        for j in range(ncols):
            row.append(gen.draw_target(img_size, target_size)[0])
        rows.append( np.hstack(row) )
    grid_img = np.vstack(rows)
    print(grid_img.shape)
    im = Image.fromarray(grid_img.astype('uint8'), 'RGB')
    im.show()
