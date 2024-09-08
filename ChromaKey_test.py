import sys
from task1 import TaskOne
from task2 import TaskTwo
def main():
    args = sys.argv[1:]
    if len(args) == 2 and args[0].startswith('-'):
        print('in task 1')
        # Task One
        color_space = args[0][1:]
        image_path = args[1]

        print(f'{color_space}: {image_path}')
        # person_with_white_bg, _ = extract_person(green_screen_img, color_space)
        # display_images(green_screen_img, person_with_white_bg, green_screen_img, green_screen_img)
        obj = TaskOne()
        obj.process(image_path, color_space)
    elif len(args) == 2:
        print('in task 2')
        # Task Two
        scenic_img = args[0]
        green_screen_img = args[1]

        # print(f'{scenic_img}: {green_screen_img}')

        obj = TaskTwo()
        obj.process(green_screen_img, scenic_img)
    elif len(args) == 1:
        print('testing all')
        # task 1 test
        if 'task1' in args:
            obj1 = TaskOne()
            obj1.test()

        # task 2 test
        elif 'task2' in args:
            obj2 = TaskTwo()
            obj2.test()

        else:
            obj1 = TaskOne()
            obj1.test()

    else:
        print("Invalid arguments.")

if __name__ == '__main__':
    main()