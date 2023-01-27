#-*-coding:utf-8-*-

selected_video_str = "0"

class SelectInterface:
    __input = 0

    def __init__(self) -> None:
        self.hello_interface()

    def hello_interface(self):
        global selected_video_str
        print(
            "选择输入视频\n" \
            "Press '1' : 近景\n" \
            "Press '2' : 远景\n" \
            "Press '3' : 有干扰的近景\n" \
            "Press '4' : 有干扰的远景\n"
        )
        self.__input = input()
            
        match self.__input:
            case '1':
                selected_video_str = 'close_range.mp4'
            case '2':
                selected_video_str = 'far_range.mp4'
            case '3':
                selected_video_str = 'interference_nearby.mp4'
            case '4':
                selected_video_str = 'interference_faraway.mp4'
            case _:
                print("输入错误\n")