# Rendering and helper functions for Carte Pole Exercise

import numpy as np
import os
import sys
import gym
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib
import matplotlib.animation as animation
import numpy as np
from PIL import Image
import time
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf


# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "rl"


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


# ----------------------------- Render Functions -----------------------------------


try:
    from pyglet.gl import gl_info

    openai_cart_pole_rendering = True  # no problem, let's use OpenAI gym's rendering function
except Exception:
    openai_cart_pole_rendering = False  # probably no X server available, let's use our own rendering function


def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,


def plot_animation(frames, repeat=False, interval=40):
    plt.close()  # or else nbagg sometimes plots in the previous cell
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    return animation.FuncAnimation(fig, update_scene, fargs=(frames, patch), frames=len(frames), repeat=repeat,
                                   interval=interval)


def plot_environment(env, figsize=(5, 4)):
    plt.close()  # or else nbagg sometimes plots in the previous cell
    plt.figure(figsize=figsize)
    img = env.render(mode="rgb_array")
    plt.imshow(img)
    plt.axis("off")
    plt.show()


# ----------------------------- Custom HUD -----------------------------------


def render_cart_pole(env, obs, step, action_val):
    if not openai_cart_pole_rendering:
        # use OpenAI gym's rendering function
        return env.render(mode="rgb_array")
    else:
        # rendering for the cart pole environment (in case OpenAI gym can't do it)
        img_w = 600
        img_h = 400
        cart_w = img_w // 12
        cart_h = img_h // 15
        pole_len = img_h // 3.5
        pole_w = img_w // 80 + 1
        x_width = 2
        max_ang = 0.2
        bg_col = (255, 255, 255)
        cart_col = 0x000000  # Blue Green Red
        pole_col = 0x669acc  # Blue Green Red

        pos, vel, ang, ang_vel = obs
        img = Image.new('RGB', (img_w, img_h), bg_col)
        draw = ImageDraw.Draw(img)
        cart_x = pos * img_w // x_width + img_w // x_width
        cart_y = img_h * 95 // 100
        top_pole_x = cart_x + pole_len * np.sin(ang)
        top_pole_y = cart_y - cart_h // 2 - pole_len * np.cos(ang)
        draw.line((0, cart_y, img_w, cart_y), fill=0)
        draw.line((cart_x, cart_y - cart_h // 2, top_pole_x, top_pole_y), fill=pole_col, width=pole_w)  # draw pole

        # --------------- Render HUD --------------------
        font = ImageFont.truetype("arial.ttf", 18)


        # draw.text((200, 200), str(action_val), font=font, fill=(255, 0, 0, 255))  # Left/right action
        # draw.text((200, 150), 'x_pos: {:0.4f}\n step:{}'
        #           .format(obs[0], step),
        #           font=font, fill=(255, 0, 0, 255))

        draw.line((cart_x, cart_y - cart_h // 2, top_pole_x, top_pole_y), fill=pole_col, width=pole_w)

        if action_val == 1:
            cart_p1 = (cart_x - cart_w // 2, cart_y - cart_h // 3)
            cart_p2 = (cart_x - cart_w // 2, cart_y + cart_h // 3)
            arrow_point = (cart_x - cart_w // 1.3, cart_y)
            draw.polygon([cart_p1, cart_p2, arrow_point], fill=(255, 0, 0))  # draw thrust arrow left
        else:
            cart_p3 = (cart_x + cart_w // 2, cart_y - cart_h // 3)
            cart_p4 = (cart_x + cart_w // 2, cart_y + cart_h // 3)
            arrow_point_r = (cart_x + cart_w // 1.3, cart_y)
            draw.polygon([cart_p3, cart_p4, arrow_point_r], fill=(255, 0, 0))  # draw thrust arrow right


        draw.rectangle((cart_x - cart_w // 2, cart_y - cart_h // 2, cart_x + cart_w // 2, cart_y + cart_h // 2),
                       fill=cart_col)  # draw cart
        draw.text((cart_x - cart_w // 2 + 22 - 4 * len(str(step)), 370), str(step), font=font,
                  fill=(255, 255, 255, 255))

        return np.array(img)






















