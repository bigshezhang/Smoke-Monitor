#-*-coding:utf-8-*-

selected_video_str = "0"

class SelectInterface:
    __input = 0

    def hello_interface(self):
        global selected_video_str
        print(
            "选择输入视频\n" \
            "Press '1' : 近景\n" \
            "Press '2' : 远景\n"
        )
        self.__input = input()
            
        if self.__input == "1" or self.__input == "2":
            if self.__input == "1":
                selected_video_str = 'close_range.mp4'
            else:
                selected_video_str = 'far_range.mp4'
        