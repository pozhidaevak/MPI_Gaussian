#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

FILE *f_size, *f_matrix, *f_vector, *f_res, *f_time;

int procQuantity;	// число доступных процессов
int rank;	// ранг текущего процесса

int* pLeadingRows; // массив для запоминания порядка выбора ведущих строк - глобальный 
int* pProcLeadingRowIter; // массив с номерами итераций, на которых строки данного процесса выбирались в качестве ведущей - локальный для каждого процесса           

int* pProcInd; // массив номеров первой строки, расположенной на процессе
int* pProcNum; // количество строк линейной системы, расположенных на процессе


// функция для выделения памяти и инициализации данных
void ProcessInitialization (double* &pVector, double* &pResult, double* &pProcRows, 
							double* &pProcVector, double* &pProcResult, int &Size, int &RowNum) 
{
  int RestRows; // оставшиеся строки - которые еще не были распределены
  int i,j;       

  if (rank == 0)
  {
/*	  
      printf("\nEnter the size of the matrix and the vector: ");
      scanf("%d", &Size);
	  */
	  
	  f_size = fopen("size.txt", "r");
	  fscanf(f_size, "%d\n", &Size);
      fclose(f_size); 
	  
  }

  MPI_Bcast(&Size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  RestRows = Size;

  for (i=0; i<rank; i++) 
    RestRows = RestRows-RestRows/(procQuantity-i);     
  RowNum = RestRows/(procQuantity-rank);

  pProcRows = (double*)malloc(RowNum*Size*sizeof(double));
  //pProcRows = new double [RowNum*Size];

  pProcVector = new double [RowNum];
  pProcResult = new double [RowNum];

  pLeadingRows = new int [Size];      
  pProcLeadingRowIter = new int [RowNum];        
  
  pProcInd = new int [procQuantity];   
  pProcNum = new int [procQuantity];  

  for (int i=0; i<RowNum; i++)   
    pProcLeadingRowIter[i] = -1;

  if (rank == 0) 
  {
    pVector = new double [Size];
    pResult = new double [Size];

	//генерация вектора правых частей системы с помощью датчика случайных чисел
	srand(unsigned(clock()));

	for (i=0; i<Size; i++)
		pVector[i] = (double) (rand() % ( 10000 - (-10000) + 1)+(-10000))/100;
  }

  // определяем, сколько строк будет храниться в каждом процессе
  // заполняем глоб. массивы pProcInd, pProcNum
  RestRows = Size;
  pProcInd[0] = 0;
  pProcNum[0] = Size/procQuantity;

  for (i=1; i<procQuantity; i++) 
  {
    RestRows -= pProcNum[i-1];
    pProcNum[i] = RestRows/(procQuantity-i);
    pProcInd[i] = pProcInd[i-1]+pProcNum[i-1];
  }

  // генерируем строки матрицы на каждом процессе
  for (i=0; i<pProcNum[rank]*Size; i++)
  {  
	pProcRows[i] = (double) (rand() % ( 10000 - (-10000) + 1)+(-10000))/100;
  }

  // рассылаем вектор правых частей системы  по процессам  
  MPI_Scatterv(pVector, pProcNum, pProcInd, MPI_DOUBLE, pProcVector, pProcNum[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);               
}

// Функция преобразования матрицы
void ColumnElimination(double* pProcRows, double* pProcVector, double* pLeadingRow, int Size, int RowNum, int Iter) 
{
  double multiplier; // множитель 

  for (int i=0; i<RowNum; i++) 
  {
    if (pProcLeadingRowIter[i] == -1) // строка еще не была ведущей
	{
      multiplier = pProcRows[i*Size+Iter] / pLeadingRow[Iter]; 

      for (int j=Iter; j<Size; j++) 
	  {
        pProcRows[i*Size + j] -= pLeadingRow[j]*multiplier;
      }

      pProcVector[i] -= pLeadingRow[Size]*multiplier;
    }
  }    
}

// Прямой ход алгоритма Гаусса
void GaussianElimination (double* pProcRows, double* pProcVector, int Size, int RowNum)
{
  double MaxValue;   // здачение ведущего элемента на данном процессе
  int    LeadingRowPos;   // позиция ведущей строки в процессе

  struct { double MaxValue; int rank; } ProcLeadingRow, LeadingRow;   // структура для ведущей строки

  // хранит ведущую строку и элемент вектора правой части
  double* pLeadingRow = new double [Size+1];

  for (int i = 0; i < Size; i++)  
  { 
    // вычисляем локальную ведущую строку
    double MaxValue = 0;             

	for (int j = 0; j < RowNum; j++) 
	{
      if ((pProcLeadingRowIter[j] == -1) && (MaxValue < fabs(pProcRows[j*Size+i]))) 
	  {
        MaxValue = fabs(pProcRows[j*Size+i]);
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
    MPI_Bcast(&pLeadingRows[i], 1, MPI_INT, LeadingRow.rank, MPI_COMM_WORLD); 
      
    if (rank == LeadingRow.rank)
	{
      // заполняем ведущую строку + записываем элемент вектора правой части
      for (int j=0; j<Size; j++) 
	  {
        pLeadingRow[j] = pProcRows[LeadingRowPos*Size + j];
      }
      pLeadingRow[Size] = pProcVector[LeadingRowPos];
    }

	// выполняем широковещательную рассылку ведущей строки и элемента вектора правой части
    MPI_Bcast(pLeadingRow, Size+1, MPI_DOUBLE, LeadingRow.rank, MPI_COMM_WORLD);

	// выполняем вычитание строк- исключаем соответствующую неизвестную
    ColumnElimination(pProcRows, pProcVector, pLeadingRow, Size, RowNum, i);
  }
}

// Функция поиска расположения ведущей строки при обратном ходе
void FindBackLeadingRow(int RowIndex, int Size, int &IterRank, int &IterLeadingRowPos) 
{
  for (int i = 0; i < procQuantity-1; i++) 
  {
	// если строка находится в ленте данного процесса 
    if ((pProcInd[i] <= RowIndex) && (RowIndex < pProcInd[i+1]))
	  IterRank = i;
  }
  if (RowIndex >= pProcInd[procQuantity-1])
    IterRank = procQuantity-1;

  // позиция строки в данном процессе
  IterLeadingRowPos = RowIndex - pProcInd[IterRank];
}

// Обратный ход алгоритма Гаусса
void BackSubstitution (double* pProcRows, double* pProcVector, double* pProcResult, int Size, int RowNum) 
{
  int IterRank;    // ранг процесса, на котором находится текущая ведущая строка
  int IterLeadingRowPos;   // позиция ведущей строки в процессе
  double IterResult;   // значение элемента результирующего вектора, вычисленное на данной итерации
  double val;

  for (int i = Size-1; i >= 0; i--) 
  {
	// ищем ведущую строку - начинаем с конца массива
	FindBackLeadingRow(pLeadingRows[i], Size, IterRank, IterLeadingRowPos);
    
    // вычисляем неизвестное
    if (rank == IterRank) 
	{
      IterResult = pProcVector[IterLeadingRowPos]/pProcRows[IterLeadingRowPos*Size+i];
	  pProcResult[IterLeadingRowPos] = IterResult;
    }

    // рассылаем полученное значение элемента результирующего вектора
    MPI_Bcast(&IterResult, 1, MPI_DOUBLE, IterRank, MPI_COMM_WORLD);

    // обновляем значения результирующего вектора 
    for (int j = 0; j < RowNum; j++) 
      if (pProcLeadingRowIter[j] < i) 
	  {
        val = pProcRows[j*Size + i] * IterResult;
        pProcVector[j]=pProcVector[j] - val;
      }
  }
}

// Функция завершения вычислений - освобождение памяти
void ProcessTermination (double* pVector, double* pResult, double* pProcRows, double* pProcVector, double* pProcResult) 
{
  if (rank == 0) 
  {
    delete [] pVector;
    delete [] pResult;
  }

  free(pProcRows);
  delete [] pProcVector;
  delete [] pProcResult;

  delete [] pLeadingRows;
  delete [] pProcLeadingRowIter;

  delete [] pProcInd;
  delete [] pProcNum;
}

int main(int argc, char* argv[]) 
{
  double* pVector;  // вектор правых частей системы
  double* pResult;  // результирующий вектор
  int	  Size;     // размер матрицы
  
  double *pProcRows;      // строки матрицы на данном процессе
  double *pProcVector;    // элементы вектора правых частей системы на данном процессе
  double *pProcResult;    // элементы результирующего вектора на данном процессе
  int     RowNum;         // количество строк матрицы, обрабатываемых данным процессом

  double  start, finish, duration; // для подсчета времени вычислений

  MPI_Init ( &argc, &argv );
  MPI_Comm_rank ( MPI_COMM_WORLD, &rank );
  MPI_Comm_size ( MPI_COMM_WORLD, &procQuantity );
  
  // выделение памяти и инициализация данных, распределение строк матрицы между процессами
  ProcessInitialization(pVector, pResult, pProcRows, pProcVector, pProcResult, Size, RowNum);

/*
  // вывод в файл исходной матрицы, сгенерированной случайным образом
  for (int i=0; i<procQuantity; i++) 
  {
	if (rank == i) 
	{
		if (rank == 0) f_matrix = fopen("matrix.txt", "w");
				else f_matrix = fopen("matrix.txt", "a+");

		for (int j=0; j<pProcNum[rank]; j++) 
		{		
			for (int k=0; k<Size; k++)
				
				printf("%7.4f ", pProcRows[j*Size+k]);
			printf("\r\n");
			  fprintf(f_matrix,"%7.4f ", pProcRows[j*Size+k]);
			fprintf(f_matrix,"\r\n");
		}
		fclose(f_matrix);
	}
	MPI_Barrier(MPI_COMM_WORLD);
  }
*/
  // включаем таймер
  start = MPI_Wtime();

  // вычисление результирующего вектора по методу Гаусса с выбором главного элемента в столбце
  // прямой ход
  GaussianElimination (pProcRows, pProcVector, Size, RowNum);

  // обратный ход
  BackSubstitution (pProcRows, pProcVector, pProcResult, Size, RowNum);
 
  // объединяем полученные на каждом процессоре результаты 
  MPI_Gatherv(pProcResult, pProcNum[rank], MPI_DOUBLE, pResult, pProcNum, pProcInd, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // останавливаем таймер
  finish = MPI_Wtime();
  duration = finish-start;
  
  // запись результатов в файлы
   
  if (rank == 0) 
  {/*
	// вывод в файл исходного вектора
	f_vector = fopen("vector.txt", "w");
	for (int i=0; i<Size; i++)
		fprintf(f_vector,"%7.4f ", pVector[i]);
	fclose(f_vector);

	// вывод результата
	f_res = fopen("result.txt", "w");
	for (int i=0; i<Size; i++)
		fprintf(f_res,"%7.4f ", pResult[pLeadingRows[i]]);
	fclose(f_res);
	*/
	// вывод значения времени, затраченного на вычисления
	f_time = fopen("time.txt", "a+");
    fprintf(f_time, " Number of processors: %d\n Size of Matrix: %d\n Time of execution: %f\n\n", procQuantity, Size, duration);
    fclose(f_time);
	/*
	  printf ("\n Result Vector: \n");
	  for (int i=0; i<Size; i++)
			printf("%7.4f ", pResult[pLeadingRows[i]]);
	  printf("\n Time of execution: %f\n", duration);
	  scanf("%d", &Size);
	*/
  }

  // Завершение процесса вычислений
  ProcessTermination(pVector, pResult, pProcRows, pProcVector, pProcResult);
  MPI_Finalize();
  return 0;
}