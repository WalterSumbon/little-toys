import curses

def draw_matrix(stdscr):
    # 初始化鼠标点击
    curses.mousemask(1)

    # 设置不回显按键
    curses.noecho()

    # 不需要按回车直接读取键值
    curses.cbreak()

    # 隐藏光标
    curses.curs_set(0)

    # 打印15x15的点阵
    for i in range(15):
        for j in range(15):
            stdscr.addch(i, j*2, '.')  # 使用两倍间距使其更像正方形点阵

    while True:
        # 刷新屏幕以显示点阵
        stdscr.refresh()

        # 等待用户事件
        event = stdscr.getch()

        if event == curses.KEY_MOUSE:
            _, x, y, _, _ = curses.getmouse()
            if x // 2 < 15 and y < 15:
                # 用户点击在点阵范围内，将该位置变为'*'
                stdscr.addch(y, x // 2 * 2, '*')
        elif event == ord('q'):  # 按'q'键退出
            break

# 包装函数以在curses环境中运行
curses.wrapper(draw_matrix)