from copy import deepcopy
import curses
import random
class Game:
    def __init__(self, board_size=(15, 15)):
        self.board_size = board_size
        self.board = [[0]*board_size[1] for _ in range(board_size[0])]  # 0: blank
        self.turn = 1   # 1: black, 2: white
        self.history = []   # list(board, choice)

    def place(self, x, y):
        if self.board[x][y] == 0:
            self.board[x][y] = self.turn

            self.history.append((deepcopy(self.board), (x, y)))

            self.turn = 3 - self.turn
            return True
        
        else:
            return False
        
    def is_over(self):
        "检查当前棋局是否已经结束，是则返回玩家代号(1/2)，不是则返回0"
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 分别对应下、右、右下、右上四个方向
        for x in range(self.board_size[0]):
            for y in range(self.board_size[1]):
                if self.board[x][y] == 0:
                    continue
                player = self.board[x][y]
                for dx, dy in directions:
                    count = 1  # 当前棋子已经算一个
                    for i in range(1, 5):  # 检查接下来的四个棋子
                        nx, ny = x + dx * i, y + dy * i
                        if 0 <= nx < self.board_size[0] and 0 <= ny < self.board_size[1] and self.board[nx][ny] == player:
                            count += 1
                        else:
                            break
                    if count == 5:
                        return player
        return 0
    
    def plot(self)->str:
        "打印棋盘状态。使用rich库"
        # 定义棋子字符
        symbols = {0: '·', 1: '●', 2: '○'}

        # 定义棋盘字符
        board_str = ''
        for i in range(self.board_size[0]):
            for j in range(self.board_size[1]):
                board_str += symbols[self.board[i][j]] + ' '
            board_str += '\n'
        return board_str
    
    def play_with_human_cli(self, player_ai, human_turn=1):
        def draw_board(stdscr):
            curses.curs_set(0)  # 隐藏光标
            stdscr.clear()  # 清屏
            stdscr.refresh()

            while True:
                # 绘制棋盘状态
                stdscr.clear()
                for i in range(self.board_size[0]):
                    for j in range(self.board_size[1]):
                        char = '·' if self.board[i][j] == 0 else '●' if self.board[i][j] == 1 else '○'
                        stdscr.addstr(i, j * 2, char)
                stdscr.refresh()

                # 检查游戏是否结束
                winner = self.is_over()
                if winner:
                    message = f"Player {winner} wins!"
                    stdscr.addstr(self.board_size[0] + 1, 0, message)
                    stdscr.refresh()
                    # stdscr.getch()  # 等待任意键退出
                    break

                # AI回合
                if self.turn != human_turn:
                    x, y = player_ai.choice(self.board, self.turn)
                    self.place(x, y)
                    continue

                # 人类玩家回合
                key = stdscr.getch()
                if key == ord('q'):  # 按q退出
                    break
                elif key == curses.KEY_MOUSE:
                    _, x, y, _, _ = curses.getmouse()
                    x, y = y, x // 2
                    if 0 <= x < self.board_size[0] and 0 <= y < self.board_size[1]:
                        if not self.place(x, y):
                            stdscr.addstr(self.board_size[0] + 1, 0, "Invalid move! Try again.")
                        else:
                            stdscr.addstr(self.board_size[0] + 1, 0, "                      ")  # 清除提示信息

        curses.wrapper(draw_board)
        
class DummyAI:
    def __init__(self):
        pass
    def choice(self, board, turn):
        candidates = []
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 0:
                    candidates.append((i, j))
        return random.choice(candidates)
    
class SillyAI:
    def __init__(self, max_distance=3):
        self.max_distance = max_distance
    def choice(self, board, turn):
        "不会远离已有的棋子超过3格"
        candidates = []
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 0:
                    for x in range(max(0, i-self.max_distance), min(len(board), i+self.max_distance+1)):
                        for y in range(max(0, j-self.max_distance), min(len(board[0]), j+self.max_distance+1)):
                            if board[x][y] != 0:
                                candidates.append((i, j))
                                break
        return random.choice(candidates)
    
class SlightlySmarterAI:
    def __init__(self):
        pass

    def choice(self, board, turn):
        """
        以下行为的优先级从高到低：
        1. 如果有一步可以赢，就走这一步
        2. 如果对手有一步可以赢，就走这一步
        3. 如果有一步可以阻止对手赢，就走这一步
        4. 如果对手有一步可以阻止我赢，就走这一步
        5. 如果有一步可以在四个方向上连成四个，就走这一步
        6. 如果对手有一步可以在四个方向上连成四个，就走这一步
        7. 如果对手有一步可以在四个方向上连成三个，就走这一步
        8. 如果自己有一步可以在四个方向上连成三个，就走这一步
        9. 下在自己的棋子周围
        10. 随机下在空位
        """
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 0:
                    for player in [turn, 3-turn]:
                        for dx, dy in directions:
                            count = 1
                            for k in range(1, 5):
                                x, y = i + dx * k, j + dy * k
                                if 0 <= x < len(board) and 0 <= y < len(board[0]) and board[x][y] == player:
                                    count += 1
                                else:
                                    break
                            if count == 5:
                                return i, j
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 0:
                    for player in [turn, 3-turn]:
                        for dx, dy in directions:
                            count = 1
                            for k in range(1, 5):
                                x, y = i + dx * k, j + dy * k
                                if 0 <= x < len(board) and 0 <= y < len(board[0]) and board[x][y] == player:
                                    count += 1
                                else:
                                    break
                            if count == 4:
                                return i, j
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 0:
                    for player in [turn, 3-turn]:
                        for dx, dy in directions:
                            count = 1
                            for k in range(1, 5):
                                x, y = i + dx * k, j + dy * k
                                if 0 <= x < len(board) and 0 <= y < len(board[0]) and board[x][y] == player:
                                    count += 1
                                else:
                                    break
                            if count == 3:
                                return i, j
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 0:
                    for player in [turn, 3-turn]:
                        for dx, dy in directions:
                            count = 1
                            for k in range(1, 5):
                                x, y = i + dx * k, j + dy * k
                                if 0 <= x < len(board) and 0 <= y < len(board[0]) and board[x][y] == player:
                                    count += 1
                                else:
                                    break
                            if count == 2:
                                return i, j
        candidates = []
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 0:
                    for x in range(max(0, i-1), min(len(board), i+2)):
                        for y in range(max(0, j-1), min(len(board[0]), j+2)):
                            if board[x][y] != 0:
                                candidates.append((i, j))
                                break
        return random.choice(candidates)

if __name__ == '__main__':
    game = Game()
    game.play_with_human_cli(SlightlySmarterAI(), human_turn=1)
    print(game.history)