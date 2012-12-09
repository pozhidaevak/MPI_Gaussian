#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

FILE *f_size, *f_matrix, *f_vector, *f_res, *f_time;

int procQuantity;	// ����� ��������� ���������
int rank;	// ���� �������� ��������

int* pLeadingRows; // ������ ��� ����������� ������� ������ ������� ����� - ���������� 
int* pProcLeadingRowIter; // ������ � �������� ��������, �� ������� ������ ������� �������� ���������� � �������� ������� - ��������� ��� ������� ��������           

int* pProcInd; // ������ ������� ������ ������, ������������� �� ��������
int* pProcNum; // ���������� ����� �������� �������, ������������� �� ��������


// ������� ��� ��������� ������ � ������������� ������
void ProcessInitialization (double* &pVector, double* &pResult, double* &pProcRows, 
							double* &pProcVector, double* &pProcResult, int &Size, int &RowNum) 
{
  int RestRows; // ���������� ������ - ������� ��� �� ���� ������������
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

	//��������� ������� ������ ������ ������� � ������� ������� ��������� �����
	srand(unsigned(clock()));

	for (i=0; i<Size; i++)
		pVector[i] = (double) (rand() % ( 10000 - (-10000) + 1)+(-10000))/100;
  }

  // ����������, ������� ����� ����� ��������� � ������ ��������
  // ��������� ����. ������� pProcInd, pProcNum
  RestRows = Size;
  pProcInd[0] = 0;
  pProcNum[0] = Size/procQuantity;

  for (i=1; i<procQuantity; i++) 
  {
    RestRows -= pProcNum[i-1];
    pProcNum[i] = RestRows/(procQuantity-i);
    pProcInd[i] = pProcInd[i-1]+pProcNum[i-1];
  }

  // ���������� ������ ������� �� ������ ��������
  for (i=0; i<pProcNum[rank]*Size; i++)
  {  
	pProcRows[i] = (double) (rand() % ( 10000 - (-10000) + 1)+(-10000))/100;
  }

  // ��������� ������ ������ ������ �������  �� ���������  
  MPI_Scatterv(pVector, pProcNum, pProcInd, MPI_DOUBLE, pProcVector, pProcNum[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);               
}

// ������� �������������� �������
void ColumnElimination(double* pProcRows, double* pProcVector, double* pLeadingRow, int Size, int RowNum, int Iter) 
{
  double multiplier; // ��������� 

  for (int i=0; i<RowNum; i++) 
  {
    if (pProcLeadingRowIter[i] == -1) // ������ ��� �� ���� �������
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

// ������ ��� ��������� ������
void GaussianElimination (double* pProcRows, double* pProcVector, int Size, int RowNum)
{
  double MaxValue;   // �������� �������� �������� �� ������ ��������
  int    LeadingRowPos;   // ������� ������� ������ � ��������

  struct { double MaxValue; int rank; } ProcLeadingRow, LeadingRow;   // ��������� ��� ������� ������

  // ������ ������� ������ � ������� ������� ������ �����
  double* pLeadingRow = new double [Size+1];

  for (int i = 0; i < Size; i++)  
  { 
    // ��������� ��������� ������� ������
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

	// ���������� ������������ ����� ���������� ������� ��������� ���������
    MPI_Allreduce(&ProcLeadingRow, &LeadingRow, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

    if (rank == LeadingRow.rank)
	{
      // ���������� ������� ������ ������� ����� 
      pProcLeadingRowIter[LeadingRowPos]= i; // ����� ��������
      pLeadingRows[i]= pProcInd[rank] + LeadingRowPos;
	}

	// ��������� ����������������� �������� ������ ������� ������   
    MPI_Bcast(&pLeadingRows[i], 1, MPI_INT, LeadingRow.rank, MPI_COMM_WORLD); 
      
    if (rank == LeadingRow.rank)
	{
      // ��������� ������� ������ + ���������� ������� ������� ������ �����
      for (int j=0; j<Size; j++) 
	  {
        pLeadingRow[j] = pProcRows[LeadingRowPos*Size + j];
      }
      pLeadingRow[Size] = pProcVector[LeadingRowPos];
    }

	// ��������� ����������������� �������� ������� ������ � �������� ������� ������ �����
    MPI_Bcast(pLeadingRow, Size+1, MPI_DOUBLE, LeadingRow.rank, MPI_COMM_WORLD);

	// ��������� ��������� �����- ��������� ��������������� �����������
    ColumnElimination(pProcRows, pProcVector, pLeadingRow, Size, RowNum, i);
  }
}

// ������� ������ ������������ ������� ������ ��� �������� ����
void FindBackLeadingRow(int RowIndex, int Size, int &IterRank, int &IterLeadingRowPos) 
{
  for (int i = 0; i < procQuantity-1; i++) 
  {
	// ���� ������ ��������� � ����� ������� �������� 
    if ((pProcInd[i] <= RowIndex) && (RowIndex < pProcInd[i+1]))
	  IterRank = i;
  }
  if (RowIndex >= pProcInd[procQuantity-1])
    IterRank = procQuantity-1;

  // ������� ������ � ������ ��������
  IterLeadingRowPos = RowIndex - pProcInd[IterRank];
}

// �������� ��� ��������� ������
void BackSubstitution (double* pProcRows, double* pProcVector, double* pProcResult, int Size, int RowNum) 
{
  int IterRank;    // ���� ��������, �� ������� ��������� ������� ������� ������
  int IterLeadingRowPos;   // ������� ������� ������ � ��������
  double IterResult;   // �������� �������� ��������������� �������, ����������� �� ������ ��������
  double val;

  for (int i = Size-1; i >= 0; i--) 
  {
	// ���� ������� ������ - �������� � ����� �������
	FindBackLeadingRow(pLeadingRows[i], Size, IterRank, IterLeadingRowPos);
    
    // ��������� �����������
    if (rank == IterRank) 
	{
      IterResult = pProcVector[IterLeadingRowPos]/pProcRows[IterLeadingRowPos*Size+i];
	  pProcResult[IterLeadingRowPos] = IterResult;
    }

    // ��������� ���������� �������� �������� ��������������� �������
    MPI_Bcast(&IterResult, 1, MPI_DOUBLE, IterRank, MPI_COMM_WORLD);

    // ��������� �������� ��������������� ������� 
    for (int j = 0; j < RowNum; j++) 
      if (pProcLeadingRowIter[j] < i) 
	  {
        val = pProcRows[j*Size + i] * IterResult;
        pProcVector[j]=pProcVector[j] - val;
      }
  }
}

// ������� ���������� ���������� - ������������ ������
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
  double* pVector;  // ������ ������ ������ �������
  double* pResult;  // �������������� ������
  int	  Size;     // ������ �������
  
  double *pProcRows;      // ������ ������� �� ������ ��������
  double *pProcVector;    // �������� ������� ������ ������ ������� �� ������ ��������
  double *pProcResult;    // �������� ��������������� ������� �� ������ ��������
  int     RowNum;         // ���������� ����� �������, �������������� ������ ���������

  double  start, finish, duration; // ��� �������� ������� ����������

  MPI_Init ( &argc, &argv );
  MPI_Comm_rank ( MPI_COMM_WORLD, &rank );
  MPI_Comm_size ( MPI_COMM_WORLD, &procQuantity );
  
  // ��������� ������ � ������������� ������, ������������� ����� ������� ����� ����������
  ProcessInitialization(pVector, pResult, pProcRows, pProcVector, pProcResult, Size, RowNum);

/*
  // ����� � ���� �������� �������, ��������������� ��������� �������
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
  // �������� ������
  start = MPI_Wtime();

  // ���������� ��������������� ������� �� ������ ������ � ������� �������� �������� � �������
  // ������ ���
  GaussianElimination (pProcRows, pProcVector, Size, RowNum);

  // �������� ���
  BackSubstitution (pProcRows, pProcVector, pProcResult, Size, RowNum);
 
  // ���������� ���������� �� ������ ���������� ���������� 
  MPI_Gatherv(pProcResult, pProcNum[rank], MPI_DOUBLE, pResult, pProcNum, pProcInd, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // ������������� ������
  finish = MPI_Wtime();
  duration = finish-start;
  
  // ������ ����������� � �����
   
  if (rank == 0) 
  {/*
	// ����� � ���� ��������� �������
	f_vector = fopen("vector.txt", "w");
	for (int i=0; i<Size; i++)
		fprintf(f_vector,"%7.4f ", pVector[i]);
	fclose(f_vector);

	// ����� ����������
	f_res = fopen("result.txt", "w");
	for (int i=0; i<Size; i++)
		fprintf(f_res,"%7.4f ", pResult[pLeadingRows[i]]);
	fclose(f_res);
	*/
	// ����� �������� �������, ������������ �� ����������
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

  // ���������� �������� ����������
  ProcessTermination(pVector, pResult, pProcRows, pProcVector, pProcResult);
  MPI_Finalize();
  return 0;
}