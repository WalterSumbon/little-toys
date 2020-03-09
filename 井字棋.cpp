#include<stdio.h>
#include<stdlib.h>
char icon[4] = " OX";
struct Node{
	int state;
	int wl;
	/* 
	<wl> win or lose
	0: O's turn
	1: X's turn
	2: O wins(first)
	3: X wins(second)
	4: Tie
	*/
	double rate_o,rate_x;
	/*
	<rate_o> the possibility of winning or tie for O
	<rate_x> the possibility of winning or tie for X
	*/
	Node *son,*bro,*best;
};
int encode(int a[3][3]){
	// 0 :
	// 1 : O
	// 2 : X
	int ans=0;
	for(int i=0;i<3;++i){
		for(int j=0;j<3;++j){
			ans = ans*3+a[i][j];
		}
	}
	return ans;
}
void decode(int state, int ans[3][3]){
	for(int i=2;i>=0;--i){
		for(int j=2;j>=0;--j){
			ans[i][j] = state%3;
			state /= 3;
		}
	}
}
void show(int state){
	int a[3][3];
	decode(state,a);
	printf("+-----+\n");
	for(int i=0;i<3;++i){
		for(int j=0;j<3;++j){
			printf("|%c",icon[a[i][j]]);
		}
		printf("|\n");
	}
	printf("+-----+\n");
}
int calc_wl(int a[3][3]){
	int cnt_o=0,cnt_x=0;
	int code_o=0,code_x=0;
	int wincode[8] = {73,146,292,448,56,7,273,84};
	for(int i=0;i<3;++i){
		for(int j=0;j<3;++j){
			code_o*=2;
			code_x*=2;
			if(a[i][j]==1){
				++code_o;
				++cnt_o;
			}
			if(a[i][j]==2){
				++code_x;
				++cnt_x;
			}
		}
	}
	for(int i=0;i<8;++i){
		if((wincode[i]&code_o) == wincode[i]){
			return 2;
		}
		if((wincode[i]&code_x) == wincode[i]){
			return 3;
		}
	}
	if(cnt_o + cnt_x == 9){
		return 4;
	}
	if(cnt_o == cnt_x){
		return 0;
	}
	return 1;
}
void calc_rate(Node* r){
	if(r->wl == 2){
		r->rate_o = 1;
		r->rate_x = 0;
		r->best = NULL;
		return ;
	}
	if(r->wl == 3){
		r->rate_o = 0;
		r->rate_x = 1;
		r->best = NULL;
		return ;
	}
	if(r->wl == 4){
		r->rate_o = 1;
		r->rate_x = 1;
		r->best = NULL;
		return;
	}
	r->rate_o = r->rate_x = 0;
	double sum = 0;
	if(r->wl == 0){
		Node *p;
		for(r->best = p = r->son; p!= NULL; p = p->bro){
			if(p->rate_o > r->best->rate_o || (p->rate_o == r->best->rate_o && p->rate_x < r->best->rate_x)){
				r->best = p;
			}
			r->rate_o += p->rate_o * p->rate_o;
			r->rate_x += p->rate_o * p->rate_x;
			sum += p->rate_o;
		}
		if(sum == 0){
			r->rate_o = 0;
			r->rate_x = 1;
			return ;
		}
	}
	if(r->wl == 1){
		Node *p;
		for(r->best = p = r->son; p!= NULL; p = p->bro){
			if(p->rate_x > r->best->rate_x || (p->rate_x == r->best->rate_x && p->rate_o < r->best->rate_o)){
				r->best = p;
			}
			r->rate_o += p->rate_x * p->rate_o;
			r->rate_x += p->rate_x * p->rate_x;
			sum += p->rate_x;
		}
		if(sum == 0){
			r->rate_x = 0;
			r->rate_o = 1;
			return ;
		}
	}
	r->rate_o /= sum;
	r->rate_x /= sum;
	return ;
}
Node* _init(int state){
	//state and wl of Node to be built
	Node *r = (Node*)malloc(sizeof(Node));
	r->state = state;
	r->bro = r->son = NULL;
	int a[3][3];
	Node* new_son;
	decode(state,a);
	r->wl = calc_wl(a);
	if(r->wl == 0 || r->wl == 1){
		for(int i=0;i<3;++i){
			for(int j=0;j<3;++j){
				if(a[i][j] == 0){
					a[i][j] = r->wl + 1;
					new_son = _init(encode(a));
					new_son->bro = r->son;
					r->son = new_son;
					a[i][j] = 0;
				}
			}
		}
	}
	calc_rate(r);
	return r;
}
Node* init(){
	return _init(0);
}
Node* place(Node* r,int x,int y){
	if(x<0||x>2||y<0||y>2){
		return r;
	}
	int a[3][3];
	decode(r->state,a);
	if(a[x][y]!=0){
		return r;
	}
	else{
		a[x][y] = r->wl + 1;
		int next_state = encode(a);
		for(Node* p = r->son;p!=NULL;p = p->bro){
			if(p->state == next_state){
				return p;
			}
		}
	}
	return NULL;
}
int choose_mode(){
	int mode;
	char input[10];
	printf("Choose your mode(o/O or x/X):\n");
	while(1){
		printf(">");
		scanf("%s",input);
		if(input[0] == 'o' || input[0] == 'O'){
			return 0;
		}
		if(input[0] == 'x' || input[0] == 'X'){
			return 1;
		}
		printf("Invalid input.Please try again:\n");
	}
}
void print_wlr(Node* r){
	double rate_t = r->rate_o + r->rate_x - 1;
	printf("\nO : %g | T : %g | X : %g\n",r->rate_o - rate_t, rate_t, r->rate_x - rate_t);
}
int main(){
	Node* tree = init();
	Node* cur;
	int mode;	//0:O	1:X
	int x,y;
	while(1){
		mode = choose_mode(); 
		cur = tree;
		print_wlr(cur);
		show(cur->state);
		while(cur){
			switch(cur->wl){
				case 0:
					if(mode == 0){
						printf(">");
						scanf("%d %d",&x,&y);
						cur = place(cur,x,y);
					}
					else{
						cur = cur->best;
					}
					break;
				case 1:
					if(mode == 1){
						printf(">");
						scanf("%d %d",&x,&y);
						cur = place(cur,x,y);
					}
					else{
						cur = cur->best;
					}
					break;
				case 2:
					printf("[O wins]\n-----<Gameover>-----\n");
					cur = NULL;
					break;
				case 3:
					printf("[X wins]\n-----<Gameover>-----\n");
					cur = NULL;
					break;
				case 4:
					printf("[Tie]\n-----<Gameover>-----\n");
					cur = NULL;
					break;
				default:
					break;
			}
			if(cur){
				print_wlr(cur);
				show(cur->state);
			}
		}
	}
	return 0;
}
