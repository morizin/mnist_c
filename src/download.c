#include <stdio.h>
#include <stdlib.h>

int main(void){
	system("curl -L -o ~/Downloads/mnist-dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/hojjatk/mnist-dataset");
	
	return 0;
}
