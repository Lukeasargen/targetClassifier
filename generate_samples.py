import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def rotate_2d(vector, angle, degrees = False):
    """
    Rotate a 2d vector counter-clockwise by @angle.\n
    vector  : [x, y]\n
    angle   : rotation angle
    degrees : set True if angle is in degrees, default is False. 
    """
    if degrees:
        angle = np.radians(angle)
    v = np.array(vector)
    c, s = np.cos(angle), np.sin(angle)
    R = np.array(((c, -s), (s, c)))
    return v.dot(R)


class TargetGenerator():
    def __init__(self):
        self.shape_options = [
            "circle", "semicircle", "quartercircle", "triangle",
            "square", "rectangle", "trapezoid", "pentagon",
            "hexagon", "heptagon", "octagon", "star", "cross"
        ]
        self.color_options = [
            'white', 'black', 'gray', 'red', 'blue',
            'green', 'yellow', 'purple', 'brown', 'orange'
        ]
        # No W or 9
        self.letter_options = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
            'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
            'X', 'Y', 'Z', '1', '2', '3', '4', '5', '6', '7', '8', '0'
        ]
        self.font_options = [
            "fonts/InputSans-Regular.ttf",
            "fonts/Lato-Black.ttf",
            "fonts/Lato-Bold.ttf",
            "fonts/Lato-Heavy.ttf",
            "fonts/Lato-Medium.ttf",
            "fonts/Lato-Regular.ttf",
            "fonts/Lato-Semibold.ttf",
            "fonts/OpenSans-Regular.ttf",
            "fonts/OpenSans-Semibold.ttf",
            "fonts/Raleway-Bold.ttf",
            "fonts/Raleway-Medium.ttf",
            "fonts/Raleway-SemiBold.ttf",
            "fonts/Roboto-Medium.ttf",
            "fonts/Roboto-Regular.ttf",
        ]
        self.angle_options = [
            "N", "NE", "E", "SE", "S", "SW", "W", "NW"
        ]

        # Reduce for debugging
        # self.shape_options = [
        #     "circle"
        # ]
        # self.color_options = [
        #     'white', 'black', 'gray', 'red', 'blue',
        # ]

        self.num_classes = [
            1,
            len(self.shape_options),
            len(self.letter_options),
            len(self.color_options),
            len(self.color_options),
        ]
        self.label_weights = None # [1.0, 1.0, 1.4, 1.0, 1.2]


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


    def make_regular_polygon(self, radius, sides, angle, center=(0,0)):
        step = 2*np.pi / sides
        points = []
        for i in range(sides):
            points.append( (radius*np.cos((step*i) + np.radians(angle))) + center[0] )
            points.append( (radius*np.sin((step*i) + np.radians(angle))) + center[1] )
        return points


    def draw_shape(self, draw, img_size, target_size, shape_idx, color,
            scale=(1.0, 1.0), rotation=False):
        shape = self.shape_options[shape_idx]
        r = (np.random.uniform(*scale)*target_size) // 2  # half width of target
        # Random center
        cx, cy = np.random.randint(r, img_size-r), np.random.randint(r, img_size-r)
        angle = 0
        if rotation:
            angle = np.random.uniform(0, 360)

        if shape == "circle":
            l = int(r*np.random.randint(50, 75) / 100)
            top = (cx-r, cy-r)
            bot = (cx+r, cy+r)
            draw.pieslice([top, bot], 0, 360, fill=color)
        elif shape == "quartercircle":
            b = np.random.choice([45, -90, -135, -180])
            # slice is in the bottom right, so shift top right rotated by angle
            rr = 2*r/np.sqrt(2)  # outer circle radius that fits the quarter circle
            ss = rr / (1+np.sqrt(2))  # inner circle radius
            l = int(ss*np.random.randint(75, 90) / 100)
            sx, sy = np.sqrt(2)*ss*np.cos(np.radians(-angle+45+b)), np.sqrt(2)*ss*np.sin(np.radians(-angle+45+b))
            top = (cx-(rr)-sx, cy-(rr)-sy)
            bot = (cx+(rr)-sx, cy+(rr)-sy)
            draw.pieslice([top, bot], 0-angle+b, 90-angle+b, fill=color)
        elif shape == "semicircle":
            # slice is in the bottom, so shift up rotated by angle
            rr = r / np.sqrt(5/4)  # outer circle radius that fits the smi circle
            l = int(0.5*rr*np.random.randint(70, 90) / 100)
            sx, sy = 0.5*rr*np.sin(np.radians(angle)), 0.5*rr*np.cos(np.radians(angle))
            top = (cx-(rr)+sx, cy-(rr)+sy)
            bot = (cx+(rr)+sx, cy+(rr)+sy)
            draw.pieslice([top, bot], 180-angle, -angle, fill=color)
        elif shape == "square":
            l = int( ( r*np.random.randint(70, 90) ) / ( 100*np.sqrt(2) ) )
            draw.regular_polygon((cx, cy, r), n_sides=4, rotation=angle, fill=color)
        elif shape == "triangle":
            radius = r*np.random.randint(85, 100) / 100
            l = int(radius*np.random.randint(40, 50) / 100)
            b = np.random.choice([30, 90])
            points = self.make_regular_polygon(radius, 3, -angle+b, center=(cx, cy))
            draw.polygon(points, fill=color)
        elif shape == "rectangle":
            h = int(r*np.random.randint(81, 97) / 100)
            w = np.sqrt(r*r - h*h)
            l = int(min(w, h)*np.random.randint(85, 96) / 100)
            b = np.random.choice([0, 90])
            points = [( +w, +h ),( +w, -h ),( -w, -h ),( -w, +h )]
            points = rotate_2d(points, angle+b, degrees=True)
            points = [(x[0]+cx, x[1]+cy) for x in points]
            draw.polygon(points, fill=color)
        elif shape == "pentagon":
            radius = r*np.random.randint(80, 100) / 100
            l = int(radius*np.random.randint(50, 75) / 100)
            b = -90/5
            points = self.make_regular_polygon(radius, 5, -angle+b, center=(cx, cy))
            draw.polygon(points, fill=color)
        elif shape == "hexagon":
            radius = r*np.random.randint(80, 100) / 100
            l = int(radius*np.random.randint(50, 80) / 100)
            points = self.make_regular_polygon(radius, 6, -angle, center=(cx, cy))
            draw.polygon(points, fill=color)
        elif shape == "heptagon":
            radius = r*np.random.randint(80, 100) / 100
            l = int(radius*np.random.randint(50, 85) / 100)
            points = self.make_regular_polygon(radius, 7, -angle, center=(cx, cy))
            draw.polygon(points, fill=color)
        elif shape == "octagon":
            radius = r*np.random.randint(80, 100) / 100
            l = int(radius*np.random.randint(50, 85) / 100)
            points = self.make_regular_polygon(radius, 8, -angle, center=(cx, cy))
            draw.polygon(points, fill=color)
        elif shape == "star":
            sides = 5
            step = 2*np.pi / sides
            b = -90  # np.random.choice([-90, 90])
            points = []
            c = r*np.random.randint(80, 100) / 100
            for i in (0, 2, 4, 1, 3):
                points.append( (c*np.cos((step*i) + np.radians(-angle+b))) + cx )
                points.append( (c*np.sin((step*i) + np.radians(-angle+b))) + cy )
            draw.polygon(points, fill=color)
            ratio = 1-np.sin(np.radians(36))*(np.tan(np.radians(18))+np.tan(np.radians(36)))
            l = int(c*ratio*np.random.randint(80, 95) / 100)
            points = self.make_regular_polygon(c*ratio, 5, -angle+b-180, center=(cx, cy))
            draw.polygon(points, fill=color)
        elif shape == "cross":
            h = int(r*np.random.randint(35, 43) / 100)
            w = np.sqrt(r*r - h*h)
            l = int(min(w, h)*np.random.randint(92, 99) / 100)
            b = np.random.choice([0, 45])
            points = [( +w, +h ),( +w, -h ),( -w, -h ),( -w, +h )]
            points1 = [(x[0]+cx, x[1]+cy) for x in rotate_2d(points, angle+b, degrees=True)]
            points2 = [(x[0]+cx, x[1]+cy) for x in rotate_2d(points, angle+90+b, degrees=True)]
            draw.polygon(points1, fill=color)
            draw.polygon(points2, fill=color)
        elif shape == "trapezoid":
            h = int(r*np.random.randint(40, 50) / 100)
            w = np.sqrt(r*r - h*h)
            o = int(w*np.random.randint(40, 70) / 100)
            l = int(min(w, h)*np.random.randint(85, 95) / 100)
            points = [( +w, +h ),( +w-o, -h ),( -w+o, -h ),( -w, +h )]
            points = [(x[0]+cx, x[1]+cy) for x in rotate_2d(points, angle, degrees=True)]
            draw.polygon(points, fill=color)

        # Draw letter circle
        t2 = (cx-l, cy-l)
        b2 = (cx+l, cy+l)
        # draw.pieslice([t2, b2], 0, 360, fill=(0,0,0))

        return (cx, cy), l, angle

    def draw_letter(self, draw, letter_size, letter_idx, color, angle):
        font_path = random.choice(self.font_options)
        font = ImageFont.truetype(font_path, size=letter_size*2)
        w, h = draw.textsize(self.letter_options[letter_idx], font=font)
        # draw.text(xy=center, text=self.letter_options[letter_idx], fill=color, font=font, anchor="mm")
        img = Image.new("RGBA", (w, h), color=(0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text((0,0), self.letter_options[letter_idx], fill=color, font=font)
        img = img.rotate(angle, expand=1)
        return img

    def draw_target(self, img_size, target_size, scale=(1.0, 1.0), rotation=False):
        # Create the label with uniform random sampling
        bkg_idx, shape_color_idx, letter_color_idx = np.random.choice(range(len(self.color_options)), 3, replace=False)
        shape_idx = np.random.randint(0, len(self.shape_options))
        letter_idx = np.random.randint(0, len(self.letter_options))

        bkg_color = self.color_to_hsv(self.color_options[bkg_idx])

        # Create a tranparent image to overlay on background images
        # img = Image.new('RGBA', size=(img_size, img_size), color=(0, 0, 0, 0))
        img = Image.new('RGBA', size=(img_size, img_size), color=bkg_color)
        draw = ImageDraw.Draw(img)

        # Render the target from the label choices
        shape_color = self.color_to_hsv(self.color_options[shape_color_idx])
        (cx, cy), l, angle = self.draw_shape(draw, img_size, target_size, shape_idx, shape_color,
            scale=scale, rotation=rotation)
        letter_angle_offset = 0
        orientation = (angle + np.random.uniform(-letter_angle_offset, letter_angle_offset)) % 360
        letter_color = self.color_to_hsv(self.color_options[letter_color_idx])
        letter_mark = self.draw_letter(draw, l, letter_idx, letter_color, orientation)
        ox, oy = letter_mark.size
        img.paste(letter_mark, (cx-ox//2, cy-oy//2), letter_mark)

        img = img.convert('RGB')
        # img = np.array(img, dtype='uint8')
        if -orientation < -180: orientation -=360
        label = {
            "orientation": -orientation/180,
            "shape": shape_idx,
            "letter": letter_idx,
            "shape_color": shape_color_idx,
            "letter_color": letter_color_idx,
        }
        return img, label


if __name__ == "__main__":

    gen = TargetGenerator()

    img_size = 64
    target_size = 60
    scale = (0.5, 1.0)
    rotation = True

    # x, y = gen.draw_target(img_size, target_size, scale=scale, rotation=rotation)
    # x.show()
    # print(y)
    # exit()

    nrows = 8
    ncols = 16
    rows = []
    for i in range(nrows):
        row = []
        for j in range(ncols):
            img, label = gen.draw_target(img_size, target_size, scale=scale, rotation=rotation)
            row.append(img)
        rows.append( np.hstack(row) )
    grid_img = np.vstack(rows)
    print(grid_img.shape)
    im = Image.fromarray(grid_img.astype('uint8'), 'RGB')
    im.show()

    # exit()

    nimg = 8000
    mean = 0.0
    var = 0.0
    for i in range(nimg):
        # img in shape [W, H, C]
        img, y = gen.draw_target(img_size, target_size, scale=scale, rotation=rotation)
        # img = np.random.normal(size=(img_size, img_size, 3)) * 255
        # [1, C, H, W], expand so that the mean function can run on dim=0
        img = np.expand_dims((np.array(img) / 255.).transpose((2, 1, 0)), axis=0)
        mean += np.mean(img, axis=(0, 2, 3))
        var += np.var(img, axis=(0, 2, 3))  # you can add var, not std
    mean = mean/nimg
    std = np.sqrt(var/nimg)
    print("mean :", mean)
    print("std :", std)
