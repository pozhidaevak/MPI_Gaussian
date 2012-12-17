#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <pthread.h>
#include <assert.h>

#define MY_RND (double)(rand() + 1) / RAND_MAX
//#define NDEBUG
//#define HARD_CODE
#ifndef NDEBUG
  #define LOG(msg, ...) printf(msg, ##__VA_ARGS__); \
  printf("\n")
  double _start;
 
  #define START() _start = MPI_Wtime();
  #define END(msg) printf(msg); printf(" Time: %f\n", MPI_Wtime() - _start);
#else
  #define LOG(msg, ...)
  #define START()
  #define END(msg)
#endif

FILE *f_size, *f_matrix, *f_vector, *f_res, *f_time;

int size;  // число доступных процессов
int BaseRowInd;
double iterResult;   // значение элемента результирующего вектора, вычисленное на данной итерации
pthread_t* threadsId;

int* pProcInd; // массив номеров первой строки, расположенной на процессе
int* pProcNum; // количество строк линейной системы, расположенных на процессе


//special C global vars
double *pVector;
double *pResult;
double *Rows;

/**
 * Инициализирует переменные, выделяет память, заполняет рандомом матрицу и вектор, вычисляет кол-во строк на каждый процессор
 * @param mSize       размерность матрицы
 */
void ProcessInitialization (
              int mSize) 
{ 
  threadsId = (pthread_t*) malloc(sizeof(pthread_t) * size);      
  //рассчитьтать кол-во и начальную строку для каждого процесса 
  pProcInd = (int*)malloc(sizeof(int) * size);   
  pProcNum = (int*)malloc(sizeof(int) * size);  
  pProcInd[0] = 0;
  pProcNum[0] = mSize / size;
  int remains = size - (mSize % size); // кол-во процессов с mSize / size строк, у остальных на одну больше
  for (int i = 1; i < remains; ++i) 
  {
    pProcNum[i] = pProcNum[0];
    pProcInd[i] = pProcInd[i - 1] + pProcNum[i - 1];
  }
  for (int i = remains; i < size; ++i)
  {
    pProcNum[i] = pProcNum[0] + 1;
    pProcInd[i] = pProcInd[i - 1] + pProcNum[i - 1];
  }

  //инициализация массивов для каждого процесса
  Rows = (double*)malloc(sizeof(double) * pProcNum[rank] * mSize); 
  

  srand(time(NULL));rand(); //на это гребанную строку ушло пол дня, которые можно бы было провести полезнее и приятние -- например плевать в потолок

  //инициализация общих для всех процессов массивов
 
    pVector = (double*)malloc(sizeof(double) * mSize);
    pResult = (double*)malloc(sizeof(double) * mSize);
  #ifndef HARD_CODE
    for (int i = 0; i < mSize; ++i)
      pVector[i] = MY_RND;
  #else
  pVector[0] = 0.393170;
  pVector[1] = 0.329722;
  pVector[2] = 0.831599;
  #endif
   #ifndef HARD_CODE
  for (int i = 0; i < mSize *mSize; ++i)
    Rows[i] = MY_RND;
   #else
    Rows[0] = 0.799280;
    Rows[1] = 0.753441;
    Rows[2] = 0.988647;
    Rows[3] = 0.481552;
    Rows[4] = 0.432295;
    Rows[5] = 0.716788;
    Rows[6] = 0.677297;
    Rows[7] = 0.181341;
    Rows[8] = 0.529557;
  #endif
  }

  //разделяем pVector между всеми
  //MPI_Scatterv(pVector, pProcNum, pProcInd, MPI_DOUBLE, pProcVector, pProcNum[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);               
}

/**
 * Осуществляет преобразование матрицы при одной базовой строке
 * @param pProcRows   строки матрицы для каждого процесса
 * @param pProcVector правая часть для каждого процесса
 * @param pBaseRow    базовая строка(с правой частью)
 * @param mSize       размерность матрицы
 * @param Iter        номер базовой строки\итерации
 */
void* ColumnElimination(void* blabla) 
{
  pthread_t self = pthread_self();
  int selfId = 0;
  
  for (int i = 0; i< size;++i)
  {
  
    if(pthread_equal(self,threadsId[i]))
    {
    
    selfId = i;
      break;
    }
  
  }
  
  double multiplier; 

  //по строкам
  for (int i = pProcInd[selfId]; i < pProcNum[selfId] + pProcInd[selfId]; ++i) 
  {
    if (BaseRowInd < i) // строка еще не была базовой
    {
      multiplier = Rows[i * mSize + BaseRowInd] / Rows[BaseRowInd * mSize + BaseRowInd]; 
      for (int j=BaseRowInd; j < mSize; ++j) 
      {
        pProcRows[i * mSize + j] -= Rows[BaseRowInd * mSize + j] * multiplier;
      }
      pVector[i] -= pVector[BaseRowInd] * multiplier;
    }
  }    
}


/**
 * Прямой ход алгоритма Гаусса
 * @param pProcRows   строки для каждого процесса
 * @param pProcVector вектор свободных коэфицентов для каждого процесса
 * @param mSize        размерность матрицы
 */
void GaussianElimination (int mSize)
{
  for (int i = 0; i < mSize; ++i)  
  {
    BaseRowInd = i;    
    //создаем нити
    for (int j = 0; j < size; ++j)
    {
      pthread_create(&threadsId[j], NULL, &ColumnElimination,NULL);
    }
    //ждем завершения
    for (int j = 0; j < size; ++j)
    {
      if(pthread_join(threadsId[j],NULL))
      {
        printf("join fails\n");fflush(stdout);
      }
    }  
  }
}

void* BackThread(void * blabla)
{
   pthread_t self = pthread_self();
  int selfId = 0;
  
  for (int i = 0; i< size;++i)
  {
  
    if(pthread_equal(self,threadsId[i]))
    { 
    selfId = i;
      break;
    }
  
  }
  // обновляем значения результирующего вектора
  double val;
  for (int j = pProcInd[selfId]; j < pProcNum[selfId] + pProcInd[selfId]; j++) 
  {
    if (j < BaseRowInd) //sign changed
    {
      val = Rows[j * mSize + BaseRowInd] * iterResult;
      pVector[j] = pVector[j] - val;
    }
  }

}

/**
 * Обратный ход алгоритма
 * @param pProcRows   сткроки для каждого процесса
 * @param pProcVector правая часть для каждого процесса
 * @param pProcResult результат для каждого процесса
 * @param mSize       размерность матрицы
 */
void BackSubstitution (int mSize) 
{
  for (int i = mSize - 1; i >= 0; --i) 
  {
    
    // вычисляем неизвестное
      iterResult = pVector[i] / Rows[i * mSize + i];
      pResult[i] = iterResult;

      BaseRowInd = i;
     for (int j = 0; j < size; ++j)
    {
      pthread_create(&threadsId[j], NULL, &BackThread,NULL);
    }
    //ждем завершения
    for (int j = 0; j < size; ++j)
    {
      if(pthread_join(threadsId[j],NULL))
      {
        printf("join fails\n");fflush(stdout);
      }
    }   
   
  }
}

// просто освобождение памяти
void ProcessTermination ()
{
  free(pVector);
  free(pResult);
  free(pProcRows);
  free(pProcInd);
  free(pProcNum);
  free(threadsId);
}

int main(int argc, char* argv[]) 
{
  int    mSize;     // размер матрицы
  double  start, finish, duration; // для подсчета времени вычислений
   
  //получить размер матрицы из коммандной строки
  if(argc < 3)
  {
    printf("No size or mSize parameter");
    return 0;
  } 
  mSize = atoi(argv[1]);
  size = atoi(argv[2]);
  
  ProcessInitialization(mSize);
  /**/
  #ifndef NDEBUG
  // вывод в файл исходной матрицы, сгенерированной случайным образом  
  f_matrix = fopen("matrix.txt", "w");
    
  for (int j=0; j < mSize; j++) 
  {    
    for (int ll=0; ll < mSize; ll++)
    {
      fprintf(f_matrix,"%f ", pProcRows[j * mSize + ll]);
    }
    fprintf(f_matrix,"\r\n");
  }
  fclose(f_matrix);
   
  #endif
  // включаем таймер
  start = (double)clock()/CLOCKS_PER_SECOND;

  //MPI_Barrier(MPI_COMM_WORLD); 
  GaussianElimination (mSize);
  //MPI_Barrier(MPI_COMM_WORLD);
  
  #ifdef HARD_CODE
  for(int i = 0; i < mSize * pProcNum[rank]; ++i)
  {
    LOG("%f", pProcRows[i]);
  }
  #endif
   
  BackSubstitution (mSize);
   
  // останавливаем таймер
  finish = (double)clock()/CLOCKS_PER_SECOND
  duration = finish-start;
 
  // запись результатов в файлы
    #ifndef NDEBUG
    // вывод в файл исходного вектора
    f_vector = fopen("vector.txt", "w");
    for (int i = 0; i < mSize; ++i)
    {
      fprintf(f_vector,"%f ", pVector[i]);
    }
    fclose(f_vector);

    // вывод результата
    f_res = fopen("result.txt", "w");
    for (int i = 0; i < mSize; ++i)
    {
      fprintf(f_res,"%f ", pResult[i]);
    }
    fclose(f_res);
    #endif
    // вывод значения времени, затраченного на вычисления
    f_time = fopen("time.txt", "a+");
    fprintf(f_time, " Number of processors: %d\n size of Matrix: %d\n Time of execution: %f\n\n", size, mSize, duration);
    printf(" Number of processors: %d\n size of Matrix: %d\n Time of execution: %f\n\n", size, mSize, duration);
    fclose(f_time);
  
  
  ProcessTermination();
  return 0;
}
