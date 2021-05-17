import imageio
import glob


def make_gif():
    anim_file = 'epochs_digits.gif'

    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob('samples/*.png')
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)
