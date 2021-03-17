import os
import random
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T  # Image processing

from custom_transforms import RandomGaussianBlur
from helper import pil_loader

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


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
    def __init__(self, input_size, target_size, expansion_factor=1, scale=(1.0, 1.0),
            rotation=False, target_transforms=None, bkg_path=None):
        """ For argument definitions, see the draw_targets() description """
        self.input_size = input_size
        self.target_size = target_size
        self.expansion_factor = expansion_factor
        self.scale = scale
        self.rotation = rotation
        self.target_transforms = target_transforms
        self.bkg_path = bkg_path
        if bkg_path:
            self.bkg_count = 0  # Track when all backgrounds are used and then reset
            # TODO : resize to save memory???
            self.backgrounds = [pil_loader(os.path.join(os.getcwd(), bkg_path, x)) for x in os.listdir(bkg_path)]  # list of pil images
            self.bkg_choices = list(range(len(self.backgrounds)))  # List of indices that is looped through to select images in self.backgrounds

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
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
            'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
            'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y',
            'Z', '1', '2', '3', '4', '5', '6', '7',
            '8', '0'
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
        self.angle_quantization = 8

        self.num_classes = {
            "orientation" : self.angle_quantization,
            "shape" : len(self.shape_options),
            "letter" : len(self.letter_options),
            "shape_color" : len(self.color_options),
            "letter_color" : len(self.color_options),
        }

    def color_to_hsv(self, color):
        """ Return a string that is used by PIL to specify HSL colorspace """
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
        """ Helper function that returns a list of tuples in a regular polygon
            which are centered at the center argument. """
        step = 2*np.pi / sides
        points = []
        for i in range(sides):
            points.append( (radius*np.cos((step*i) + np.radians(angle))) + center[0] )
            points.append( (radius*np.sin((step*i) + np.radians(angle))) + center[1] )
        return points

    def draw_shape(self, draw, input_size, target_size, shape_idx, color,
            scale=(1.0, 1.0), rotation=False):
        """ Do not use. this is called within draw_target.
            This function draws the specified shape, color, scale, and rotation.
            It returns values that specify how to draw the letter.
        """
        shape = self.shape_options[shape_idx]
        r = (np.random.uniform(*scale)*target_size) // 2  # half width of target
        # Random center
        cx, cy = np.random.randint(r, input_size-r), np.random.randint(r, input_size-r)
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
            radius = r*np.random.randint(85, 100) / 100
            l = int( ( r*np.random.randint(70, 90) ) / ( 100*np.sqrt(2) ) )
            b = np.random.choice([0, 45])
            points = self.make_regular_polygon(radius, 4, -angle+b, center=(cx, cy))
            draw.polygon(points, fill=color)
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
            b = -90/5 + np.random.choice([0, 180])
            points = self.make_regular_polygon(radius, 5, -angle+b, center=(cx, cy))
            draw.polygon(points, fill=color)
        elif shape == "hexagon":
            radius = r*np.random.randint(80, 100) / 100
            l = int(radius*np.random.randint(50, 80) / 100)
            b = np.random.choice([0, 30])
            points = self.make_regular_polygon(radius, 6, -angle+b, center=(cx, cy))
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
            c = r*np.random.randint(94, 100) / 100
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

        # return center, letter_size, angle in degrees
        return (cx, cy), l, angle

    def draw_letter(self, draw, letter_size, letter_idx, color, angle):
        """ Do not use. this is called within draw_target.
            This function chooses a random font a draws the
            specified letter on a transparent PIL image. This
            image has the specified size, color, and angle."""
        font_path = random.choice(self.font_options)
        font = ImageFont.truetype(font_path, size=letter_size*2)  # Double since letter_size is based on radius
        w, h = draw.textsize(self.letter_options[letter_idx], font=font)
        # draw.text(xy=center, text=self.letter_options[letter_idx], fill=color, font=font, anchor="mm")
        img = Image.new("RGBA", (w, h), color=(0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text((0,0), self.letter_options[letter_idx], fill=color, font=font)
        img = img.rotate(angle, expand=1)
        return img

    def draw_target(self, input_size=None, target_size=None, scale=None, rotation=None,
            target_transforms=None, bkg_path=None):
        """ Draws a random target on a transparent PIL image. Also returns the correct labels.
            
            input_size:
                square size in pixels
            target_size:
                maximum diameter of circle that the target could fit inside
            scale:
                tuple likes this (0.5, 1.0), upper and lower bounds of the target_size. sampled uniformly
            rotation:
                boolean true or false. The orientation label is returned regardless.
            target_transforms:
                function that gets called on the final image before returning. 
                It is best to just us the pytorch transforms or your own functions that work on PIL images
            bkg_path:
                if not None, then a target with transparent background is returned. else, a random
                color is selected that is different from the shape and letter color
        """
        if input_size == None:
            input_size = self.input_size
        if target_size == None:
            target_size = self.target_size
        if scale == None:
            scale = self.scale
        if rotation == None:
            rotation = self.rotation
        if target_transforms == None:
            target_transforms = self.target_transforms

        # Create the label with uniform random sampling
        bkg_color_idx, shape_color_idx, letter_color_idx = np.random.choice(range(len(self.color_options)), 3, replace=False)
        shape_idx = np.random.randint(0, len(self.shape_options))
        letter_idx = np.random.randint(0, len(self.letter_options))

        # Create a tranparent image to overlay on background images
        target = Image.new('RGBA', size=(input_size*self.expansion_factor, input_size*self.expansion_factor), color=(0, 0, 0, 0))
        draw = ImageDraw.Draw(target)

        # Render the target from the label choices
        shape_color = self.color_to_hsv(self.color_options[shape_color_idx])
        (cx, cy), l, angle = self.draw_shape(draw, input_size*self.expansion_factor, target_size*self.expansion_factor, shape_idx, shape_color,
            scale=scale, rotation=rotation)
        letter_angle_offset = 180 if rotation else 0 # Allows the letter to not aligned with how a regular person would draw a letter
        angle = (angle + np.random.uniform(-letter_angle_offset, letter_angle_offset)) % 360
        letter_color = self.color_to_hsv(self.color_options[letter_color_idx])
        letter_mark = self.draw_letter(draw, l, letter_idx, letter_color, angle)
        ox, oy = letter_mark.size
        temp = Image.new('RGBA', size=(input_size*self.expansion_factor, input_size*self.expansion_factor), color=(0, 0, 0, 0))
        temp.paste(letter_mark, (cx-(ox//2), cy-(oy//36)-(oy//2)), letter_mark)  # Put letter with offset based on size
        target.alpha_composite(temp)  # removes the transparent aliasing border from the letter

        if target_transforms:
            target = target_transforms(target)

        if bkg_path == None:
            bkg_color = self.color_to_hsv(self.color_options[bkg_color_idx])
            img = Image.new('RGB', size=(input_size*self.expansion_factor, input_size*self.expansion_factor), color=bkg_color)
            img.paste(target, None, target)  # Alpha channel is the mask
        else:
            img = target  # return the target with transparency since it will be pasted on the background

        # TODO : make angle label for hyperbolic network
        angle = angle % 360

        # Quantize angle in cw direction, turns orientation into a classification task
        # mod by quantization to avoid error at 0 degrees (rotation=False)
        offset =   360 / self.angle_quantization
        orientation = int(np.floor((360-angle-offset)*(self.angle_quantization)/360)) % self.angle_quantization

        label = {
            "angle" : angle,
            "orientation": orientation,
            "shape": shape_idx,
            "letter": letter_idx,
            "shape_color": shape_color_idx,
            "letter_color": letter_color_idx,
        }
        return img, label

    def get_background(self):
        """ Works like a iterator that randomly samples the images loaded
            from the bkg_path folder. The list is PIL images in ram. It
            returns full resolution PIL image"""
        self.bkg_count += 1
        if self.bkg_count == len(self.backgrounds):
            np.random.shuffle(self.bkg_choices)
            self.bkg_count = 0
        return self.backgrounds[self.bkg_choices[self.bkg_count]]            

    def gen_classify(self, input_size=None, target_size=None, scale=None, rotation=None,
            target_transforms=None, bkg_path=None):
        """ Generate a cropped target with it's classification label.
            For argument explainations see the draw_targets() description."""
        if input_size == None:
            input_size = self.input_size
        if bkg_path == None:
            bkg_path = self.bkg_path
        target, label = self.draw_target(input_size=input_size, target_size=target_size, scale=scale,
                            rotation=rotation, target_transforms=target_transforms, bkg_path=bkg_path)
        if bkg_path:
            bkg = self.get_background()
            img = T.RandomResizedCrop(input_size*self.expansion_factor, scale=(0.08, 1.0), ratio=(3./4., 4./3.), interpolation=Image.NEAREST)(bkg)
            img.paste(target, None, target)  # Alpha channel is the mask
        else:
            img = target.convert("RGB")  # return the target

        img = img.resize((input_size, input_size))
        return img, label

    def gen_segment(self, input_size=None, target_size=None, fill_prob=None,
            target_transforms=None, bkg_path=None):
        """ Generate an aerial image with it's target mask.
            
            input_size:
                the final size of the image, can be a single value (square) or tuple (width, height)
            target_size:
                the smallest size of the targets. the largest target will be based on the 
                input_size. targets can cover up to 25% of the image

            """
        if input_size == None:
            input_size = self.input_size
        if target_size == None:
            target_size = self.target_size
        if fill_prob == None:
            fill_prob = 1.0
        if bkg_path == None:
            bkg_path = self.bkg_path

        # pick random gridsize based on input and target_size
        if isinstance(input_size, int):
            bkg_w, bkg_h = input_size, input_size
        elif isinstance(input_size, Sequence) and len(input_size) == 2:
            bkg_w, bkg_h = input_size
        # print("bkg_w, bkg_h :", bkg_w, bkg_h)

        scale_w, scale_h = bkg_w//target_size, bkg_h//target_size  # Smallest grid cells based on the smallest target
        # print("scale_w, scale_h :", scale_w, scale_h)
        
        max_num = min(scale_w, scale_h)
        num = np.random.randint(2, max_num) if max_num>2 else 1 # Divisions along the smallest dimension
        # print("num :", num)

        # Scale divisions two both axis
        if bkg_w > bkg_h:
            num_w = int(scale_w/scale_h * num)
            num_h = num
        else:
            num_h = int(scale_h/scale_w * num)
            num_w = num
        # print("num_w, num_h :", num_w, num_h)

        step_w, step_h = bkg_w/num_w, bkg_h/num_h  # Rectangle size for each target
        # print("step_w, step_h :", step_w, step_h)

        new_target_size = int(min(step_w, step_h))  # Largest target that can fit
        # print("new_target_size :", new_target_size)

        new_scale = (1.0, target_size/new_target_size)  # New scale between the largest target and the smallest target
        # print("new_scale :", new_scale)

        # This mask is first used to place all the targets, then converted into a binary image
        place_targets = Image.new('RGBA', size=(bkg_w*self.expansion_factor, bkg_h*self.expansion_factor), color=(0, 0, 0, 0))

        # fill the grid randomly with targets
        target_prob_mask = np.random.rand(num_w*num_h) < fill_prob
        # print("target_prob_mask :", target_prob_mask)

        offset_x, offset_y = int(step_w-new_target_size), int(step_h-new_target_size)
        for i in range(len(target_prob_mask)):
            y = i // num_w
            x = (i - y*num_w)
            if target_prob_mask[i]:
                # print("x, y :", x, y, int(x*step_w), int(y*step_h))
                target, label = self.draw_target(input_size=new_target_size, target_size=new_target_size-1, scale=new_scale, rotation=True,
                                    target_transforms=target_transforms, bkg_path=True)
                # draw_image makes a sqaure image, so it can be shifted in the rectangle
                ox = np.random.randint(0, offset_x) if offset_x != 0 else 0
                oy = np.random.randint(0, offset_y) if offset_y != 0 else 0
                place_targets.paste(target, (int((x*step_w+ox)*self.expansion_factor), int((y*step_h+oy)*self.expansion_factor)), target)  # Alpha channel is the mask

        mask = Image.new('RGBA', size=(bkg_w*self.expansion_factor, bkg_h*self.expansion_factor), color=(0, 0, 0, 0))
        mask.alpha_composite(place_targets)  # removes the transparent aliasing border from the mask

        if bkg_path:
            bkg = self.get_background()
            img = T.RandomResizedCrop(size=(bkg_h*self.expansion_factor, bkg_w*self.expansion_factor), scale=(0.08, 1.0), ratio=(3./4., 4./3.), interpolation=Image.NEAREST)(bkg)
            img.paste(mask, None, mask)  # Alpha channel is the mask
        else:
            img = mask.convert("RGB")  # return the mask in rgb

        img = img.resize((bkg_w, bkg_h))

        # Convert the transparency to the binary mask
        # temp = Image.new('RGB', size=(bkg_w*self.expansion_factor, bkg_h*self.expansion_factor), color=(0, 0, 0))
        # temp.paste(Image.fromarray(255*np.ones((bet.size[1], bet.size[0]))), None, mask)
        mask = Image.fromarray(np.asarray(mask)[:,:,3])
        mask = mask.convert("RGB").resize((bkg_w, bkg_h))

        return img, mask


def visualize_classify(gen):
    nrows = 8
    ncols = 8
    rows = []
    for i in range(nrows):
        row = []
        for j in range(ncols):
            img, label = gen.gen_classify()
            row.append(img)
        rows.append( np.hstack(row) )
    grid_img = np.vstack(rows)
    print(grid_img.shape)
    im = Image.fromarray(grid_img.astype('uint8'), 'RGB')
    im.show()
    im.save("images/high_res_targets.jpeg")


if __name__ == "__main__":

    bkg_path = 'backgrounds'  # path to background images
    input_size = 96
    target_size = 94
    scale = (0.8, 1.0)
    rotation = True
    expansion_factor = 3  # generate higher resolution targets and downscale, improves aliasing effects

    target_tranforms = T.Compose([
        T.RandomPerspective(distortion_scale=0.4, p=1.0, interpolation=Image.BICUBIC),
    ])

    # create the generator object
    # this can be used for classification and segmentation generation
    gen = TargetGenerator(input_size=input_size, target_size=target_size, expansion_factor=expansion_factor,
        scale=scale, rotation=rotation, target_transforms=target_tranforms, bkg_path=bkg_path)
    
    # x, y = gen.gen_classify()
    # x.show()
    # print(y)

    # visualize_classify(gen)

    img, mask = gen.gen_segment(input_size=(256, 256), target_size=20, fill_prob=0.5)
    out = Image.fromarray(np.hstack([np.asarray(img), np.asarray(mask)]))
    out.show()
    # out.save("images/seg_example.png")