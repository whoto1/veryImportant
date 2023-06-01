/*Реализация модели нейронной сети с тремя слоями "fully connected" для классификации данных.
Входные данные считываются из файла, проходят через слои нейронной сети, 
и выводится результат классификации. Код использует библиотеку cuBLAS для ускорения вычислений на GPU.*/

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <math.h>
#include <cublas_v2.h>

char ERROR_WITH_WEIGHT_FILE[] = "Error writing in weight file\n";
char ERROR_WITH_INPUT_FILE[] = "Error writing in input file\n";

cublasHandle_t handle;

// Функция сигмоиды
__global__ void sigm(float* x) {
    int idx = threadIdx.x;
    x[idx] = exp(x[idx]) / (1 + exp(x[idx]));
}

// Реализация "fully connected" слоя
class Linear {
    float* weight;
    float* bias;
    int in_features;
    int out_features;

public:
    Linear() {
        weight = NULL;
        bias = NULL;
        in_features = 0;
        out_features = 0;
    };

    Linear(int in, int out) {
        weight = NULL;
        bias = NULL;
        in_features = in;
        out_features = out;
    }

    // Инициализируем "weights" и "bias"
    void initializer(FILE* weights){
        float* w = (float*)malloc(in_features * out_features * sizeof(float));
        float* b = (float*)malloc(out_features * sizeof(float));

        fread(w, sizeof(float), in_features*out_features, weights);
        fread(b, sizeof(float), out_features, weights);
	
	////////////////////////////////////////
	//Выделение памяти на GPU для массивов weight и bias.
        cudaMalloc((void**)&weight, in_features * out_features * sizeof(float));
        cudaMalloc((void**)&bias, out_features * sizeof(float));
	    
	//Копирование данных из w и b на хосте в weight и bias на устройстве соответственно.
        cudaMemcpy(weight, w, in_features * out_features * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(bias, b, out_features * sizeof(float), cudaMemcpyHostToDevice);
	////////////////////////////////////////
	    
        free(w);
        free(b);
    }

    // Вектор с входными данными умножается на матрицу весов
    float* operator() (float* x) {
        const float a = 1;
	    
	// Умножаю матрицы weight (транспонированной) на вектор x
	// с добавлением bias и сохранением результата в x
        cublasSgemv(handle, CUBLAS_OP_T, in_features, out_features, &a, weight, in_features, x, 1, &a, bias, 1);
	    
	// Копируем вектор bias в x
        cublasScopy(handle, out_features, bias, 1, x, 1);  
        return x;
    }
    ~Linear() {
        if (weight) cudaFree(weight);
        if (bias) cudaFree(bias);
    }
};

// Модель нейронной сети с тремя fc слоями
class Net {
    Linear fc1;
    Linear fc2;
    Linear fc3;

    // Прямое распространение информации
    float forward(float* x) {
        sigm<<<1, 256>>>(fc1(x));
        sigm<<<1, 16>>>(fc2(x));
        sigm<<<1, 1>>>(fc3(x));

        float result;
	// Копирую значения из переменной x на устройстве в переменную result на хосте
        cudaMemcpy(&result, x, sizeof(float), cudaMemcpyDeviceToHost);
        return result;
    }
public:
    Net(int in, int middle1, int middle2) {
	// Создаю handle для работы с библиотекой cuBLAS
        cublasCreate(&handle);

        FILE* weight = fopen("weight.npy", "rb");

        if (weight == NULL) {
            printf(ERROR_WITH_WEIGHT_FILE);
            exit(1);
        }

        fc1 = Linear(in, middle1);
        fc2 = Linear(middle1, middle2);
        fc3 = Linear(middle2, 1);

        fc1.initializer(weight);
        fc2.initializer(weight);
        fc3.initializer(weight);
    }

    // Запуск нейронной сети. 
    //Чтение входных данных из файла и запуск прямого потока информации
    float operator() (char* file, int size) {        
        FILE* input = fopen(file, "rb");
        if (input == NULL) {
            printf(ERROR_WITH_INPUT_FILE);
            exit(1);
        }

        float* input_layer = (float*)malloc(size * sizeof(float));  
    
        if(input_layer){
            fread(input_layer, sizeof(float), size, input);
        }

        float* d_layer;
	// Выделяю память на GPU для массива d_layer размером size*sizeof(float)
        cudaMalloc((void**)&d_layer, size*sizeof(float));
	    
	// Копирую данные из input_layer на хосте в d_layer на устройстве.
        cudaMemcpy(d_layer, input_layer, size*sizeof(float), cudaMemcpyHostToDevice);
	free(input_layer);
        
        return forward(d_layer);
    }
    ~Net(){
        cublasDestroy(handle);
    }
};

// The main function
int main() {
    char input_file[] = "input.npy";

    int size = 1024;

    Net neuralNetwork = Net(1024, 256, 16);    

    float result = neuralNetwork(input_file, size);
    printf(">>> Result: %lf\n", result);    

    return 0;
}
