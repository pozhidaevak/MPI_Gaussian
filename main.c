#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>
#include <assert.h>

#define MY_RND (double)(rand() + 1) / RAND_MAX
//#define NDEBUG
#define NOMPI
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
int rank;  // ранг текущего процесса

//int* pLeadingRows; // массив для запоминания порядка выбора ведущих строк - глобальный 
//int* pProcLeadingRowIter; // массив с номерами итераций, на которых строки данного процесса выбирались в качестве ведущей - локальный для каждого процесса           

int* pProcInd; // массив номеров первой строки, расположенной на процессе
int* pProcNum; // количество строк линейной системы, расположенных на процессе


// функция для выделения памяти и инициализации данных
void ProcessInitialization (double* &pVector, double* &pResult, double* &pProcRows, 
              double* &pProcVector, double* &pProcResult, int mSize) 
{
  int i,j;       

  //рассчитьтать кол-во и начальную строку для каждого процесса TODO may be MPI_SCatter?
  pProcInd = (int*)malloc(sizeof(int) * size);   
  pProcNum = (int*)malloc(sizeof(int) * size);  
  pProcInd[0] = 0;
  pProcNum[0] = mSize / size;
  int remains = size - (mSize % size); // кол-во процессов с mSize / size строк, у остальных на одну больше
  for (i = 1; i < remains; ++i) 
  {
    pProcNum[i] = pProcNum[0];
    pProcInd[i] = pProcInd[i - 1] + pProcNum[i - 1];
  }
  for (i = remains; i < size; ++i)
  {
    pProcNum[i] = pProcNum[0] + 1;
    pProcInd[i] = pProcInd[i - 1] + pProcNum[i - 1];
  }

  //инициализация массивов для каждого процесса
  pProcRows = (double*)malloc(sizeof(double) * pProcNum[rank] * mSize); //TODO may be double**?
  pProcVector = (double*)malloc(sizeof(double) * pProcNum[rank]);
  pProcResult = (double*)malloc(sizeof(double) * pProcNum[rank]);

  //pLeadingRows = (int*)malloc(sizeof(int) * mSize);      
  //pProcLeadingRowIter = (int*)malloc(sizeof(int) * pProcNum[rank]);        
  
  //for (int i=0; i<pProcNum[rank]; i++)   
  //  pProcLeadingRowIter[i] = -1;
srand(time(NULL)+ rank);
  if (!rank) 
  {
    pVector = (double*)malloc(sizeof(double) * mSize);
    pResult = (double*)malloc(sizeof(double) * mSize);

    //генерация вектора правых частей системы с помощью датчика случайных чисел
	
    for (i = 0; i < mSize; i++)
      pVector[i] = MY_RND;
	
  }

  // генерируем строки матрицы на каждом процессе
  for (i = 0; i < pProcNum[rank] * mSize; ++i)
  {  
    pProcRows[i] = MY_RND;
  }

  // рассылаем вектор правых частей системы  по процессам
  MPI_Scatterv(pVector, pProcNum, pProcInd, MPI_DOUBLE, pProcVector, pProcNum[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);               
}

// Функция преобразования матрицы
void ColumnElimination(double* pProcRows, double* pProcVector, double* pLeadingRow, int mSize, int Iter) 
{
  double multiplier; // множитель 

  //по строкам
  for (int i=0; i < pProcNum[rank]; i++) 
  {
    if (Iter < pProcInd[rank] + i) // строка еще не была ведущей
    {
      multiplier = pProcRows[i * mSize + Iter] / pLeadingRow[Iter]; 

      for (int j=Iter; j < mSize; j++) 
      {
        pProcRows[i*mSize + j] -= pLeadingRow[j] * multiplier;
      }

      pProcVector[i] -= pLeadingRow[mSize] * multiplier;
    }
  }    
}

// Функция поиска расположения ведущей строки при обратном ходе
void RowIndToRankAndOffset(int rowInd, int mSize, int &iterRank, int &iterOffset) 
{
  assert(rowInd < mSize);
  iterRank = -1;
  for (int i = 1; i < size; ++i) 
  {
  // если строка находится в строках данного процесса 
    if (rowInd < pProcInd[i])
    {
      iterRank = i - 1;
      break;
    }
  }
  if (rowInd >= pProcInd[size - 1])
    iterRank = size - 1;

  assert(iterRank >= 0);
  // позиция строки в данном процессе
  iterOffset = rowInd - pProcInd[iterRank];
}

// Прямой ход алгоритма Гаусса
/**
 * [GaussianElimination description]
 * @param pProcRows   строки для каждого процесса
 * @param pProcVector вектор свободных коэфицентов для каждого процесса
 * @param mSize        размерность матрицы
 */
void GaussianElimination (double* pProcRows, double* pProcVector, int mSize)
{
  double* pLeadingRow = (double*)malloc(sizeof(double) * (mSize + 1));
  /*double MaxValue;   // здачение ведущего элемента на данном процессе
  int    LeadingRowPos;   // позиция ведущей строки в процессе

  struct { double MaxValue; int rank; } ProcLeadingRow, LeadingRow;   // структура для ведущей строки

  // хранит ведущую строку и элемент вектора правой части
  
*/
  for (int i = 0; i < mSize; i++)  
  {
  /* 
    // вычисляем локальную ведущую строку
    double MaxValue = 0;             

    for (int j = 0; j < pProcNum[rank]; j++) 
    {
        if ((pProcLeadingRowIter[j] == -1) && (MaxValue < fabs(pProcRows[j*mSize+i]))) 
        {
          MaxValue = fabs(pProcRows[j*mSize+i]);
          LeadingRowPos = j; 
        }
    }
    ProcLeadingRow.MaxValue = MaxValue;
    ProcLeadingRow.rank = rank;

    // определяем максимальный среди полученных ведущих элементов процессов
    MPI_Allreduce(&ProcLeadingRow, &LeadingRow, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
    
    if (rank == LeadingRow.rank)
    {
      // запоминаем порядок выбора ведущих строк 
      pProcLeadingRowIter[LeadingRowPos]= i; // номер итерации
      pLeadingRows[i]= pProcInd[rank] + LeadingRowPos;
    }

    // выполняем широковещательную рассылку номера ведущей строки   
    MPI_Bcast(&pLeadingRows[i], 1, MPI_INT, LeadingRow.rank, MPI_COMM_WORLD);*/ 
	
    //Вычисляем ранг и смещение катой строки
    int leadingRowRank;
    int leadingRowPos;
    RowIndToRankAndOffset(i, mSize, leadingRowRank, leadingRowPos);
	
    if (rank == leadingRowRank)
    {
      // заполняем ведущую строку + записываем элемент вектора правой части
      for (int j = 0; j < mSize; ++j) 
      {
        pLeadingRow[j] = pProcRows[leadingRowPos * mSize + j];
      }
      pLeadingRow[mSize] = pProcVector[leadingRowPos];
    }
	
    // выполняем широковещательную рассылку ведущей строки и элемента вектора правой части
    MPI_Bcast(pLeadingRow, mSize + 1, MPI_DOUBLE, leadingRowRank, MPI_COMM_WORLD);
	
    // выполняем вычитание строк- исключаем соответствующую неизвестную

    ColumnElimination(pProcRows, pProcVector, pLeadingRow, mSize, i);
	
    
  }
  free(pLeadingRow);
}



// Обратный ход алгоритма Гаусса
void BackSubstitution (double* pProcRows, double* pProcVector, double* pProcResult, int mSize) 
{
  int iterRank;    // ранг процесса, на котором находится текущая ведущая строка
  int IterLeadingRowPos;   // позиция ведущей строки в процессе
  double iterResult;   // значение элемента результирующего вектора, вычисленное на данной итерации
  double val;

  for (int i = mSize - 1; i >= 0; --i) 
  {
    RowIndToRankAndOffset(i, mSize, iterRank, IterLeadingRowPos);
    
    // вычисляем неизвестное
    if (rank == iterRank) 
    {
      iterResult = pProcVector[IterLeadingRowPos] / pProcRows[IterLeadingRowPos * mSize + i];
      pProcResult[IterLeadingRowPos] = iterResult;
    }

    // рассылаем полученное значение элемента результирующего вектора
    MPI_Bcast(&iterResult, 1, MPI_DOUBLE, iterRank, MPI_COMM_WORLD);

    // обновляем значения результирующего вектора 
    for (int j = 0; j < pProcNum[rank]; j++) 
      if (pProcNum[rank] + j > i) //sign changed
      {
        val = pProcRows[j*mSize + i] * iterResult;
        pProcVector[j]=pProcVector[j] - val;
      }
  }
}

// Функция завершения вычислений - освобождение памяти
void ProcessTermination (double* &pVector, double* &pResult, double* &pProcRows, double* &pProcVector, double* &pProcResult) 
{
  if (!rank) 
  {
     free(pVector);
     free(pResult);
  }
  free(pProcRows);
  free(pProcVector);
  free(pProcResult);
  //free(pLeadingRows);
  //free(pProcLeadingRowIter);
  free(pProcInd);
  free(pProcNum);
}

int main(int argc, char* argv[]) 
{
  double* pVector;  // вектор правых частей системы
  double* pResult;  // результирующий вектор
  int    mSize;     // размер матрицы
  
  double *pProcRows;      // строки матрицы на данном процессе
  double *pProcVector;    // элементы вектора правых частей системы на данном процессе
  double *pProcResult;    // элементы результирующего вектора на данном процессе

  double  start, finish, duration; // для подсчета времени вычислений

  MPI_Init ( &argc, &argv );
  MPI_Comm_rank ( MPI_COMM_WORLD, &rank );
  MPI_Comm_size ( MPI_COMM_WORLD, &size );
    
  

  //получить размер матрицы из коммандной строки
  if(argc < 2)
  {
    printf("No size parameter");
    MPI_Finalize();
    return 0;
  }
  if(!rank)
  {
    mSize = atoi(argv[1]);
  }
  MPI_Bcast(&mSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // выделение памяти и инициализация данных, распределение строк матрицы между процессами
  ProcessInitialization(pVector, pResult, pProcRows, pProcVector, pProcResult, mSize);
  /**/
  // вывод в файл исходной матрицы, сгенерированной случайным образом
  for (int i=0; i<size; i++) 
  {
  if (rank == i) 
  {
    if (!rank) f_matrix = fopen("matrix.txt", "w");
        else f_matrix = fopen("matrix.txt", "a+");

    for (int j=0; j<pProcNum[rank]; j++) 
    {    
      for (int ll=0; ll<mSize; ll++)
      {
        
        fprintf(f_matrix,"%7.4f ", pProcRows[j*mSize + ll]);
      }
      fprintf(f_matrix,"\r\n");
    }
    fclose(f_matrix);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  }
  /**/
  // включаем таймер
  start = MPI_Wtime();

  // вычисление результирующего вектора по методу Гаусса с выбором главного элемента в столбце
  // прямой ход
  LOG("Starting elimination time: %f",MPI_Wtime() - start);
  GaussianElimination (pProcRows, pProcVector, mSize);
 
  // обратный ход

  LOG("Starting substitution time: %f",MPI_Wtime() - start);
  BackSubstitution (pProcRows, pProcVector, pProcResult, mSize);
  LOG("Starting Gaterf %f",MPI_Wtime() - start);
  // объединяем полученные на каждом процессоре результаты 
  MPI_Gatherv(pProcResult, pProcNum[rank], MPI_DOUBLE, pResult, pProcNum, pProcInd, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   
  // останавливаем таймер
  finish = MPI_Wtime();
  duration = finish-start;
 
  
  // запись результатов в файлы
   
  if (!rank) 
  {
	  /**/
  // вывод в файл исходного вектора
  f_vector = fopen("vector.txt", "w");
  for (int i=0; i<mSize; i++)
    fprintf(f_vector,"%7.4f ", pVector[i]);
  fclose(f_vector);

  // вывод результата
  f_res = fopen("result.txt", "w");
  for (int i=0; i<mSize; i++)
    fprintf(f_res,"%7.4f ", pResult[i]);
  fclose(f_res);
  /**/
  // вывод значения времени, затраченного на вычисления
  f_time = fopen("time.txt", "a+");
    fprintf(f_time, " Number of processors: %d\n size of Matrix: %d\n Time of execution: %f\n\n", size, mSize, duration);
	printf(" Number of processors: %d\n size of Matrix: %d\n Time of execution: %f\n\n", size, mSize, duration);
    fclose(f_time);
  
   
  
  }
  
  ProcessTermination(pVector, pResult, pProcRows, pProcVector, pProcResult);
  MPI_Finalize();
  return 0;
}