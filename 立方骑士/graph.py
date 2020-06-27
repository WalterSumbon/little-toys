from copy import deepcopy
import os
class Node(object):
    def __init__(self,coord,face_idx,id,isstart = False,isend = False):
        self.neighbors = set()
        self.visited = False
        self.coord = coord
        self.face_idx = face_idx
        self.isstart = isstart
        self.isend = isend
        self.sync_set = {self}
        self.idx = 0
        self.id = id     #大于等于2时表示全局唯一的同步标号
    def add_nb(self,a):
        for n in self.sync_set:
            n.neighbors.add(a)
    def visit(self,idx):
        for n in self.sync_set:
            n.visited = True
            n.idx = idx
    def unvisit(self):
        for n in self.sync_set:
            n.visited = False
    def add_sync(self,a):
        neighbors = self.neighbors | a.neighbors
        sync_set = self.sync_set | a.sync_set
        for n in self.sync_set:
            n.neighbors = neighbors
            n.sync_set = sync_set
        for n in a.sync_set:
            n.neighbors = neighbors
            n.sync_set = sync_set
        
class Board(object):
    def __init__(self,face_idx,matrix):
        """a为边长。matrix中1为需要经过的格子，2为起点，3为终点，0为空白，其他为全局唯一的同步标号"""
        self.a = len(matrix)
        self.matrix = deepcopy(matrix)
        self.face_idx = face_idx
        self.nodes = []
        self.sync_dict = {}
        for i in range(self.a):
            for j in range(self.a):
                id = matrix[i][j]
                if id > 0:
                    self.nodes.append(Node((i,j),face_idx,id,isstart = (id == 2),isend = (id == 3)))
                if id > 1:
                    self.sync_dict[id] = self.nodes[-1]
        def is_nb(a,b):
            m = abs(a.coord[0]-b.coord[0])
            n = abs(a.coord[1]-b.coord[1])
            if abs(m-n) == 1 and m+n == 3:
                return True
            return False
        for i in range(len(self.nodes)):
            for j in range(i+1,len(self.nodes)):
                if is_nb(self.nodes[i],self.nodes[j]):
                    self.nodes[i].add_nb(self.nodes[j])
                    self.nodes[j].add_nb(self.nodes[i])
    def show(self):
        pic = self.matrix
        for n in self.nodes:
            pic[n.coord[0]][n.coord[1]] = n.idx
        print("[%d]"%(self.face_idx))
        for i in range(self.a):
            for j in range(self.a):
                if pic[i][j]:
                    print("%3d"%pic[i][j],end='|')
                else:
                    print(' '*3,end = '|')
            print('\n'+"---+"*self.a)
    @classmethod
    def adjust(cls,a,b,m,n):
        def _edge(a,m):
            if m == 0:
                return [(i,0,a.face_idx) for i in range(a.a)]
            if m == 1:
                return [(7,i,a.face_idx) for i in range(a.a)]
            if m == 2:
                return [(i,7,a.face_idx) for i in range(a.a-1,-1,-1)]
            if m == 3:
                return [(0,i,a.face_idx) for i in range(a.a-1,-1,-1)]
        e_a = _edge(a,m)
        e_b = _edge(b,n)
        return [(i,j) for i,j in zip(e_a,e_b[::-1])]

class Graph(object):
    def __init__(self,boards,pares = ()):   #pares用来手动合并点
        self.boards = boards
        self.board_dict = {}
        for b in boards:
            self.board_dict[b.face_idx] = b

        def find_node(x,y,f):
            board = self.board_dict[f]
            for n in board.nodes:
                if n.coord[0] == x and n.coord[1] == y:
                    return n
            return None
                

        def sync(pare):
            a = find_node(pare[0][0], pare[0][1],pare[0][2])
            b = find_node(pare[1][0], pare[1][1],pare[1][2])
            if a != None and b != None:
                a.add_sync(b)   #今后所有操作a,b同步

        for pare in pares:
            sync(pare)
        #根据标号合并
        for i in range(len(self.boards)):
            for j in range(i+1, len(self.boards)):
                m,n = self.boards[i].sync_dict,self.boards[j].sync_dict
                for id in m.keys():
                    if id in n.keys():
                        m[id].add_sync(n[id])

        self.nodes = [n for b in boards for n in b.nodes]
        self.node_count = 0
        for n in self.nodes:
            if n.visited == False:
                n.visit(0)
                self.node_count += 1
        for n in self.nodes:
            if n.isstart:
                self.start = n
            if n.isend:
                self.end = n
            n.unvisit()
    def show(self):
        for b in self.boards:
            b.show()
    
    def _dfs(self,a,count):
        count += 1
        a.visit(count)
        # if count >= 57:
        #     print(count,a.coord,a.face_idx)
        if count == self.node_count:
            for board in self.boards:
                board.show()
            return True
        elif a.id != 3:
            for n in a.neighbors:
                if not n.visited and self._dfs(n,count):
                    return True
        a.unvisit()
        count -= 1
        return False

    def dfs(self):
        return self._dfs(self.start,0)

    def compile(self,filename,log_file_name=None):
        code_head = r"""#include <stdio.h>"""
        code_node_definition = r"""
        struct Node{
            int id;
            int idx;
            int x,y,f;
            bool visited = false;
            int nb_count = 0;    //the number of neighbors
            int alive_nb_count;
            Node *nbs[8];
            void add_nb(Node *p){
                nbs[nb_count++] = p;
            }
            bool visit(int &single,int &target){   //return value: is this a wrong way
                visited = true;
                bool isolated = false;
                single = 0;
                target = -1;
                for(int i=0; i<nb_count; ++i){
                    --(nbs[i]->alive_nb_count);
                    if(nbs[i]->id != 3){
                        if(nbs[i]->alive_nb_count < 1 && !(nbs[i]->visited)){
                            isolated = true;
                        }
                        else if(nbs[i]->alive_nb_count == 1 && !(nbs[i]->visited)){
                            ++single;
                            target = i;
                        }
                    }
                }
                if(single > 1){
                    return true;
                }
                return isolated;
            }
            void unvisit(){
                visited = false;
                for(int i=0; i<nb_count; ++i){
                    ++(nbs[i]->alive_nb_count);
                }
            }
        };
        """
        code_declare_globs = "Node g[%d];\nconst int N = %d;\n"%(self.node_count,self.node_count)

        nodes = []
        count = 0
        for n in self.nodes:        #give every node an 'idx', which declares its index in 'nodes'
            if not n.visited:
                n.visit(count)
                nodes.append(n)
                count+=1

        code_init_nodes = ''
        code_add_nb = []
        for i,n in enumerate(nodes):
            code_init_nodes += ''.join(["g[%d].id = %d;\n"%(i,n.id),
                                        "g[%d].idx = %d;\n"%(i,i),
                                        "g[%d].x = %d;\n"%(i,n.coord[0]),
                                        "g[%d].y = %d;\n"%(i,n.coord[1]),
                                        "g[%d].f = %d;\n"%(i,n.face_idx)])
            for m in list(n.neighbors):
                code_add_nb.append("g[%d].add_nb(&g[%d]);\n"%(i,m.idx))
            code_add_nb.append("g[%d].alive_nb_count = g[%d].nb_count;\n"%(i,i))
        code_add_nb = ''.join(code_add_nb)

        code_dfs = r"""
        int max = 0;
        bool dfs(Node *p,int count){
            count += 1;
            int single,target;
            if(p->visit(single,target)){
                p->unvisit();
                return false;
            }
            p->idx = count;
            if(count > max){
            	max = count;
            	printf("%d (%d,%d) %d\n",count,p->x,p->y,p->f);
            }
            if(count == N){
                show();
                return true;
            }
            else if(p->id != 3){
                if(single != 1){
                    for(int i = 0; i < p->nb_count;++i){
                        if(!(p->nbs[i]->visited) && dfs(p->nbs[i],count)){
                            return true;
                        }
                    }
                }
                else{   //single == 1 : There is only one way to go
                    if(!(p->nbs[target]->visited) && dfs(p->nbs[target],count)){
                            return true;
                        }
                }
            }
            p->unvisit();
            return false;
        }
        """
        code_show = r"""
        void show(){
            int b[6][8][8];
            for(int i = 0; i <6;++i){
                for(int j = 0; j <8;++j){
                    for(int k = 0; k <8;++k){
                        b[i][j][k] = 0;
                    }
                }
            }
            for(int i=0;i<N;++i){
                int x,y,f;
                b[g[i].f][g[i].x][g[i].y] = g[i].idx;
                
            }
            for(int f = 0; f < 6;++f){
                printf("\n[%d]\n",f);
                for(int i = 0; i < 8;++i){
                    for(int j = 0; j < 8;++j){
                        if(b[f][i][j]==0){
                            printf("   |");
                        }
                        else{
                            printf("%3d|",b[f][i][j]);
                        }
                    }
                    printf("\n---+---+---+---+---+---+---+---+\n");
                }
            }
        }
        """
        code_log = r"""
        void log(const char* filename){
            FILE * f = fopen(filename,"w");
            for(int i = 0; i <N; ++i){
                fprintf(f,"%d ",g[i].idx);
            }
            fclose(f);
        }
        """
        code_main_function = "int main(){\n"+\
                             code_init_nodes+code_add_nb+\
                             'printf("build complete!\\n");\n'+\
                             'dfs(&g[%d],0);\n'%(self.start.idx)+\
                             '//show();\n'+\
                             ('log("%s");'%(log_file_name) if log_file_name else '')+\
                             'return 0;}\n'
        with open(filename,'w') as f:
            f.write(code_head)
            f.write(code_node_definition)
            f.write(code_declare_globs)
            f.write(code_show)
            f.write(code_dfs)
            if(log_file_name):
                f.write(code_log)
            f.write(code_main_function)
        os.system("gcc %s -o %s"%(filename,filename[:-4]))
        os.system("./%s > /dev/null"%filename[:-4])
        if(log_file_name):
            with open(log_file_name,'r') as f:
                idx_list = f.readline()
                idx_list = [int(i) for i in idx_list.split(' ')[:-1]]
                for i,n in enumerate(nodes):
                    n.visit(idx_list[i])
                self.show()
                