#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>
#include <assert.h>

#define MY_RND (double)(rand() + 1) / RAND_MAX
//#define NDEBUG
#define HARD_CODE
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

int* pProcInd; // массив номеров первой строки, расположенной на процессе
int* pProcNum; // количество строк линейной системы, расположенных на процессе


//special C global vars
double *pVector;
double *pResult;
double *pProcRows;
double *pProcVector;
double *pProcResult;

/**
 * Инициализирует переменные, выделяет память, заполняет рандомом матрицу и вектор, вычисляет кол-во строк на каждый процессор
 * @param pVector     вектор(правая часть)
 * @param pResult     результирующий вектор
 * @param pProcRows   Набор строк для каждого процесса
 * @param pProcVector часть вектора для каждого процесса
 * @param pProcResult часть результата для каждого процесса
 * @param mSize       размерность матрицы
 */
void ProcessInitialization (
              int mSize) 
{       
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
  pProcRows = (double*)malloc(sizeof(double) * pProcNum[rank] * mSize); 
  pProcVector = (double*)malloc(sizeof(double) * pProcNum[rank]);
  pProcResult = (double*)malloc(sizeof(double) * pProcNum[rank]);

  srand(time(NULL) + 2 * rank);rand(); //на это гребанную строку ушло пол дня, которые можно бы было провести полезнее и приятние -- например плевать в потолок

  //инициализация общих для всех процессов массивов
  if (!rank) 
  {
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

  }

  for (int i = 0; i < pProcNum[rank] * mSize; ++i)
  {
	#ifndef HARD_CODE
    pProcRows[i] = MY_RND;
	#else
	if (size == 3)
	{
		switch (rank)
		{
		case 0:
			pProcRows[0] = 0.799280;
			pProcRows[1] = 0.753441;
			pProcRows[2] = 0.988647;
			break;
		case 1:
			pProcRows[0] = 0.481552;
			pProcRows[1] = 0.432295;
			pProcRows[2] = 0.716788;
			break;
		case 2:
			pProcRows[0] = 0.677297;
			pProcRows[1] = 0.181341;
			pProcRows[2] = 0.529557;
			break;

		}
	}
	else
	{
		pProcRows[0] = 0.799280;
		pProcRows[1] = 0.753441;
		pProcRows[2] = 0.988647;
		pProcRows[3] = 0.481552;
		pProcRows[4] = 0.432295;
		pProcRows[5] = 0.716788;
		pProcRows[6] = 0.677297;
			pProcRows[7] = 0.181341;
			pProcRows[8] = 0.529557;
	}
	#endif
  }

  //разделяем pVector между всеми
  MPI_Scatterv(pVector, pProcNum, pProcInd, MPI_DOUBLE, pProcVector, pProcNum[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);               
}

/**
 * Осуществляет преобразование матрицы при одной базовой строке
 * @param pProcRows   строки матрицы для каждого процесса
 * @param pProcVector правая часть для каждого процесса
 * @param pBaseRow    базовая строка(с правой частью)
 * @param mSize       размерность матрицы
 * @param Iter        номер базовой строки\итерации
 */
void ColumnElimination(double* pBaseRow, int mSize, int Iter) 
{
  double multiplier; 

  //по строкам
  for (int i = 0; i < pProcNum[rank]; ++i) 
  {
    if (Iter < pProcInd[rank] + i) // строка еще не была базовой
    {
      multiplier = pProcRows[i * mSize + Iter] / pBaseRow[Iter]; 
      for (int j=Iter; j < mSize; ++j) 
      {
        pProcRows[i * mSize + j] -= pBaseRow[j] * multiplier;
      }
      pProcVector[i] -= pBaseRow[mSize] * multiplier;
    }
  }    
}

/**
 * Определяет процесс в котором находится строка и смещение по абсолютному номеру строки
 * @param rowInd     Абсолютный номер строки
 * @param mSize      Размерность матрицы
 * @param iterRank   Номер процесса к которому принадлежит строка
 * @param iterOffset смещение строки в процессе
 */
void RowIndToRankAndOffset(int rowInd, int mSize, int* iterRank, int *iterOffset) 
{
  assert(rowInd < mSize);
 *iterRank = -1;
  for (int i = 1; i < size; ++i) 
  {
    if (rowInd < pProcInd[i])
    {
      *iterRank = i - 1;
      break;
    }
  }
  if (rowInd >= pProcInd[size - 1])
    *iterRank = size - 1;

  assert(*iterRank >= 0);
  // смещение строки в данном процессе
  *iterOffset = rowInd - pProcInd[*iterRank];
  //LOG("Offset calc: i = %d; rank = %d; offset = %d",rowInd, *iterRank, *iterOffset);
}

/**
 * Прямой ход алгоритма Гаусса
 * @param pProcRows   строки для каждого процесса
 * @param pProcVector вектор свободных коэфицентов для каждого процесса
 * @param mSize        размерность матрицы
 */
void GaussianElimination (int mSize)
{
  double* pBaseRow = (double*)malloc(sizeof(double) * (mSize + 1));
  for (int i = 0; i < mSize; ++i)  
  {  	
    //Вычисляем ранг и смещение итой строки
    int baseRowRank;
    int baseRowPos;
    RowIndToRankAndOffset(i, mSize, &baseRowRank, &baseRowPos);
	//MPI_Barrier(MPI_COMM_WORLD);
    if (rank == baseRowRank)
    {
	//LOG("Base rowRank %d; offset %d", rank, baseRowPos);
      // заполняем ведущую строку + записываем элемент вектора правой части
      for (int j = 0; j < mSize; ++j) 
      {
        pBaseRow[j] = pProcRows[baseRowPos * mSize + j];
      }
      pBaseRow[mSize] = pProcVector[baseRowPos];
	  //LOG(" BaseRow 0: %f BaseRow mSize: %f rank:%d pos:%d", pBaseRow[0], pBaseRow[mSize], rank, baseRowPos);
    }
	//MPI_Barrier(MPI_COMM_WORLD);
    // передаем базовую строку
    MPI_Bcast(pBaseRow, mSize + 1, MPI_DOUBLE, baseRowRank, MPI_COMM_WORLD);
	if (!rank)
	{
		//LOG("BaseRow 0: %f BaseRow mSize: %f", pBaseRow[0], pBaseRow[mSize]);
	}
	//MPI_Barrier(MPI_COMM_WORLD);
    ColumnElimination(pBaseRow, mSize, i); 
	MPI_Barrier(MPI_COMM_WORLD);  
  }
  free(pBaseRow);
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
  int iterRank;    // номер процесса, на котором находится текущая базовая строка
  int iterBaseRowPos;   // смещение базовой строки в процессе
  double iterResult;   // значение элемента результирующего вектора, вычисленное на данной итерации
  double val;

  for (int i = mSize - 1; i >= 0; --i) 
  {
    RowIndToRankAndOffset(i, mSize, &iterRank, &iterBaseRowPos);
    
    // вычисляем неизвестное
    if (rank == iterRank) 
    {
      iterResult = pProcVector[iterBaseRowPos] / pProcRows[iterBaseRowPos * mSize + i];
      pProcResult[iterBaseRowPos] = iterResult;
    }

    // рассылаем полученное значение результата
    MPI_Bcast(&iterResult, 1, MPI_DOUBLE, iterRank, MPI_COMM_WORLD);

    // обновляем значения результирующего вектора 
    for (int j = 0; j < pProcNum[rank]; j++) 
      if (pProcInd[rank] + j < i) //sign changed
      {
        val = pProcRows[j * mSize + i] * iterResult;
        pProcVector[j] = pProcVector[j] - val;
      }
  }
}

// просто освобождение памяти
void ProcessTermination ()
{
  if (!rank) 
  {
    free(pVector);
    free(pResult);
  }
  free(pProcRows);
  free(pProcVector);
  free(pProcResult);
  free(pProcInd);
  free(pProcNum);
}

int main(int argc, char* argv[]) 
{
  int    mSize;     // размер матрицы
  

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
  
  ProcessInitialization(mSize);
  /**/
  #ifndef NDEBUG
  // вывод в файл исходной матрицы, сгенерированной случайным образом
  for (int i=0; i<size; ++i) 
  {
    if (rank == i) 
    {
      if (!rank)
      {  
        f_matrix = fopen("matrix.txt", "w");
      }
      else
      { 
        f_matrix = fopen("matrix.txt", "a+");
      }

      for (int j=0; j<pProcNum[rank]; j++) 
      {    
        for (int ll=0; ll<mSize; ll++)
        {
          fprintf(f_matrix,"%f ", pProcRows[j*mSize + ll]);
        }
        fprintf(f_matrix,"\r\n");
      }
      fclose(f_matrix);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  #endif
  // включаем таймер
  start = MPI_Wtime();

  MPI_Barrier(MPI_COMM_WORLD); 
  GaussianElimination (mSize);
  MPI_Barrier(MPI_COMM_WORLD);
  
  #ifdef HARD_CODE
  for(int i = 0; i < mSize * pProcNum[rank]; ++i)
  {
	  LOG("%f", pProcRows[i]);
  }
	#endif
   
 
  BackSubstitution (mSize);
  MPI_Barrier(MPI_COMM_WORLD);
  
  // объединяем полученные на каждом процессоре результаты 
  MPI_Gatherv(pProcResult, pProcNum[rank], MPI_DOUBLE, pResult, pProcNum, pProcInd, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   
  // останавливаем таймер
  finish = MPI_Wtime();
  duration = finish-start;
 
  // запись результатов в файлы
  if (!rank) 
  {
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
  }
  
  ProcessTermination();
  MPI_Finalize();
  return 0;
}
