main.py:

The following code is present at the bottom of the file, in "if __name__ == '__main__':"

r_shift, g_shift = main('path/to/input/image','path/to/output/image')
print(f'R: {r_shift}')
print(f'G: {g_shift}')

Simply replace the 'path/to/input/image' and 'path/to/output/image' appropriately.
The aligned image will be saved to the output path and the best (x,y) shifts will be displayed



photo_editing.py:

The following code is present at the bottom of the file, in "if __name__ == '__main__':"

input_path = 'path/to/input/image'
output_paths = {'0_1': 'path/to/0_1/output',
                'hist_eq': 'path/to/hist_eq/output',
                'ad_hist_eq': 'path/to/ad_hist_eq/output',
                'gray_world': 'path/to/gray_world/output',
                'avg_world': 'path/to/avg_world/output'}

main(input_path, output_paths)

Simply replace the 'path/to/input/image' and all the output paths appropriately
