#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/*
 *  This is my realisation of conversion for 3 matrixes with binary input.
 *  WARNING: Structure use only unsigned data, BUT in conversion functions core (D-matrix) casts as int8_t
 *  OPTIMIZATION: i could use optimization for vert+hor instead full conv. for any rank < n/2, but it usefull only
 *  if i know the meaning of core-D, so i build optimization as example and not so slow version for 0,1 rank
 *  optimization.
 */


/**
 * @brief enum for int-returns
 */
enum ERROR
{
    ERROR_NONE = 0,
    WRONG_INPUT = 10,
    MEMORY_ALLOCATION_ERROR,
    BAD_ARGS = -1,
};

/**
 * @brief Matrix structure with 8byte width/len and unsigned 1-byte data
 */
typedef struct
{
    uint8_t* data;
    uint32_t width;
    uint32_t height;
} Matrix;

/**
 * @brief allocates a matrix data and pointer itself
 * @param p return Matrix pointer (changes it)
 * @param w Matrix width
 * @param d Matrix height
 * @return Error code (ERROR_NONE if none, MEMORY_ALLOCATION_ERROR if bad alloc)
 */
int matrix_alloc(Matrix** p, int w, int d)
{
    *p = malloc(sizeof(Matrix));
    if (*p == NULL)
        return MEMORY_ALLOCATION_ERROR;
    (*p)->width = w;
    (*p)->height = d;
    (*p)->data = (uint8_t*)calloc(d * w, sizeof(uint8_t));
    if ((*p)->data == NULL) return MEMORY_ALLOCATION_ERROR;
    return ERROR_NONE;
}

/**
 * @brief Frees a matrix
 * @param p addres of Matrix pointer (change it and data inside)
 * @return Error code
 */
int matrix_free(Matrix** p)
{
    if (p == NULL) return BAD_ARGS;
    if (*p == NULL) return BAD_ARGS;
    free((*p)->data);
    (*p)->data = NULL;
    free(*p);
    *p = NULL;
    return ERROR_NONE;
}

/**
 * @brief fills a matrix with zeros
 * @param p Matrix pointer (changes it)
 * @return Error code
 */
int fill_zero(Matrix* p)
{
    for (uint32_t i = 0; i < p->width * p->height; i++)
    {
        p->data[i] = 0;
    }
    return ERROR_NONE;
}

/**
 * @brief checks the matrix rank
 * @param m Matrix pointer
 * @return 0 for 0 matrix, 1 for rank 1, -1 otherwise
 * @warning ONLY 0, 1, -1
 */
int rank_of(Matrix* m)
{
    if (m == NULL || m->data == NULL)
        return BAD_ARGS;

    uint32_t h = m->height;
    uint32_t w = m->width;

    if (h == 0 || w == 0)
        return BAD_ARGS;

    uint32_t ref_row = 0;
    uint32_t ref_col = 0;
    int found = 0;

    for (uint32_t i = 0; i < h * w; i++)
    {
        if ((int8_t)m->data[i] != 0)
        {
            ref_row = i / w;
            ref_col = i % w;
            found = 1;
            break;
        }
    }

    if (!found)
        return 0;

    int64_t ref =
        (int8_t)m->data[ref_row * w + ref_col];

    for (uint32_t i = 0; i < h; i++)
    {
        int64_t left =
            (int8_t)m->data[i * w + ref_col];

        for (uint32_t j = 0; j < w; j++)
        {
            int64_t value =
                (int8_t)m->data[i * w + j];

            int64_t top =
                (int8_t)m->data[ref_row * w + j];

            if (value * ref != left * top)
                return -1;
        }
    }

    return 1;
}

/**
 * @brief Performs standard 2D convolution (but as i know it not rotate core - so it is correlation)
 * @param A_p (return value inside) First input matrix
 * @param B_p (return value inside)  Second input matrix
 * @param C_p (return value inside)  Third input matrix
 * @param core Convolution core
 * @return Error code
 * @warning CORE CAST AS SIGNED INT8!!!
 */
int basic_3_conversion(Matrix** A_p, Matrix** B_p, Matrix** C_p, Matrix* core)
{
    if (A_p == NULL || B_p == NULL || C_p == NULL || core == NULL)
        return BAD_ARGS;

    if (*A_p == NULL || *B_p == NULL || *C_p == NULL)
        return BAD_ARGS;

    Matrix* A = *A_p;
    Matrix* B = *B_p;
    Matrix* C = *C_p;

    const int wb = A->width;
    const int hb = A->height;

    const int wc = core->width;
    const int hc = core->height;

    Matrix* A1 = NULL;
    Matrix* B1 = NULL;
    Matrix* C1 = NULL;

    if (matrix_alloc(&A1, wb, hb))
    {
        return MEMORY_ALLOCATION_ERROR;
    }

    if (matrix_alloc(&B1, wb, hb))
    {
        matrix_free(&A1);
        return MEMORY_ALLOCATION_ERROR;
    }

    if (matrix_alloc(&C1, wb, hb))
    {
        matrix_free(&A1);
        matrix_free(&B1);
        return MEMORY_ALLOCATION_ERROR;
    }

    int sumA;
    int sumB;
    int sumC;
    int x, y;
    const int pref_w = wc / 2;
    const int pref_h = hc / 2;
    int id_base, id_core;
    for (int j = 0; j < hb; j++)
    {
        for (int i = 0; i < wb; i++)
        {
            sumA = 0;
            sumB = 0;
            sumC = 0;

            for (int h = 0; h < hc; h++)
            {
                for (int w = 0; w < wc; w++)
                {
                    x = i + w - pref_w;
                    y = j + h - pref_h;

                    if (x < 0 || x >= wb ||
                        y < 0 || y >= hb)
                    {
                        continue;
                    }

                    id_base = y * wb + x;
                    id_core = h * wc + w;
                    sumA += A->data[id_base] * (int8_t)(core->data[id_core]);
                    sumB += B->data[id_base] * (int8_t)core->data[id_core];
                    sumC += C->data[id_base] * (int8_t)core->data[id_core];
                }
            }

            if (sumA > 255)
                sumA %= 251;
            else if (sumA < 0)
                sumA = (-sumA) % 241;

            if (sumB > 255)
                sumB %= 251;
            else if (sumB < 0)
                sumB = (-sumB) % 241;

            if (sumC > 255)
                sumC %= 251;
            else if (sumC < 0)
                sumC = (-sumC) % 241;

            id_base = j * wb + i;

            A1->data[id_base] = (uint8_t)sumA;
            B1->data[id_base] = (uint8_t)sumB;
            C1->data[id_base] = (uint8_t)sumC;
        }
    }

    matrix_free(A_p);
    matrix_free(B_p);
    matrix_free(C_p);

    *A_p = A1;
    *B_p = B1;
    *C_p = C1;

    return ERROR_NONE;
}

/**
 * @brief Performs optimized convolution for a rank-1 core for 3 matrixes (also without core rotation)
 * @param A_p (return value inside) First input matrix
 * @param B_p  (return value inside) Second input matrix
 * @param C_p (return value inside) Third input matrix
 * @param core Rank-1 convolution core
 * @return Error code
 * @warning CORE CAST AS SIGNED INT8!!!
 */
int conversion_3_by_1(Matrix** A_p, Matrix** B_p, Matrix** C_p, Matrix* core)
{
    if (A_p == NULL || B_p == NULL || C_p == NULL || core == NULL)
        return BAD_ARGS;

    if (*A_p == NULL || *B_p == NULL || *C_p == NULL)
        return BAD_ARGS;

    Matrix* A = *A_p;
    Matrix* B = *B_p;
    Matrix* C = *C_p;

    const uint32_t wb = A->width;
    const uint32_t hb = A->height;

    const uint32_t wc = core->width;
    const uint32_t hc = core->height;

    uint32_t ref_row = 0;
    uint32_t ref_col = 0;
    int found = 0;

    for (uint32_t y = 0; y < hc && !found; y++)
    {
        for (uint32_t x = 0; x < wc; x++)
        {
            if ((int8_t)core->data[y * wc + x] != 0)
            {
                ref_row = y;
                ref_col = x;
                found = 1;
                break;
            }
        }
    }

    if (!found)
        return BAD_ARGS;

    const int64_t pivot =
        (int8_t)core->data[ref_row * wc + ref_col];

    int64_t* tmpA = calloc((size_t)wb * hb, sizeof(int64_t));
    int64_t* tmpB = calloc((size_t)wb * hb, sizeof(int64_t));
    int64_t* tmpC = calloc((size_t)wb * hb, sizeof(int64_t));

    if (tmpA == NULL || tmpB == NULL || tmpC == NULL)
    {
        free(tmpA);
        free(tmpB);
        free(tmpC);
        return MEMORY_ALLOCATION_ERROR;
    }

    const int pref_w = wc / 2;
    const int pref_h = hc / 2;

    for (uint32_t y = 0; y < hb; y++)
    {
        for (uint32_t x = 0; x < wb; x++)
        {
            int64_t sumA = 0;
            int64_t sumB = 0;
            int64_t sumC = 0;

            for (uint32_t h = 0; h < hc; h++)
            {
                int src_y = (int)y + (int)h - pref_h;

                if (src_y < 0 || src_y >= (int)hb)
                    continue;

                const int64_t v =
                    (int8_t)core->data[h * wc + ref_col];

                uint32_t id = (uint32_t)src_y * wb + x;

                sumA += A->data[id] * v;
                sumB += B->data[id] * v;
                sumC += C->data[id] * v;
            }

            uint32_t id = y * wb + x;

            tmpA[id] = sumA;
            tmpB[id] = sumB;
            tmpC[id] = sumC;
        }
    }

    Matrix* A1 = NULL;
    Matrix* B1 = NULL;
    Matrix* C1 = NULL;

    if (matrix_alloc(&A1, wb, hb))
    {
        free(tmpA);
        free(tmpB);
        free(tmpC);
        return MEMORY_ALLOCATION_ERROR;
    }

    if (matrix_alloc(&B1, wb, hb))
    {
        matrix_free(&A1);
        free(tmpA);
        free(tmpB);
        free(tmpC);
        return MEMORY_ALLOCATION_ERROR;
    }

    if (matrix_alloc(&C1, wb, hb))
    {
        matrix_free(&A1);
        matrix_free(&B1);
        free(tmpA);
        free(tmpB);
        free(tmpC);
        return MEMORY_ALLOCATION_ERROR;
    }

    for (uint32_t y = 0; y < hb; y++)
    {
        for (uint32_t x = 0; x < wb; x++)
        {
            int64_t sumA = 0;
            int64_t sumB = 0;
            int64_t sumC = 0;

            for (uint32_t w = 0; w < wc; w++)
            {
                int src_x = (int)x + (int)w - pref_w;

                if (src_x < 0 || src_x >= (int)wb)
                    continue;

                const int64_t h =
                    (int8_t)core->data[ref_row * wc + w];

                uint32_t id = y * wb + (uint32_t)src_x;

                sumA += tmpA[id] * h;
                sumB += tmpB[id] * h;
                sumC += tmpC[id] * h;
            }

            sumA /= pivot;
            sumB /= pivot;
            sumC /= pivot;

            if (sumA > 255)
                sumA %= 251;
            else if (sumA < 0)
                sumA = (-sumA) % 241;

            if (sumB > 255)
                sumB %= 251;
            else if (sumB < 0)
                sumB = (-sumB) % 241;

            if (sumC > 255)
                sumC %= 251;
            else if (sumC < 0)
                sumC = (-sumC) % 241;

            uint32_t id = y * wb + x;

            A1->data[id] = (uint8_t)sumA;
            B1->data[id] = (uint8_t)sumB;
            C1->data[id] = (uint8_t)sumC;
        }
    }

    free(tmpA);
    free(tmpB);
    free(tmpC);

    matrix_free(A_p);
    matrix_free(B_p);
    matrix_free(C_p);

    *A_p = A1;
    *B_p = B1;
    *C_p = C1;

    return ERROR_NONE;
}

/**
* @brief Reads matrices from a binary file
* @param input Input file path in c-str
* @param res Output matrix array
* @return Error code
*/
int read_input(const char* input, Matrix*** res)
{
    FILE* fp = fopen(input, "rb");
    if (fp == NULL)
        return WRONG_INPUT;

    uint32_t height;
    uint32_t width;

    if (fread(&height, sizeof(height), 1, fp) != 1)
        goto error_input;

    if (fread(&width, sizeof(width), 1, fp) != 1)
        goto error_input;

    Matrix* A;
    Matrix* B;
    Matrix* C;

    if (matrix_alloc(&A, width, height) != ERROR_NONE)
        goto error_alloc;

    if (matrix_alloc(&B, width, height) != ERROR_NONE)
    {
        matrix_free(&A);
        goto error_alloc;
    };

    if (matrix_alloc(&C, width, height) != ERROR_NONE)
    {
        matrix_free(&A);
        matrix_free(&B);
        goto error_alloc;
    }

    for (uint32_t i = 0; i < width * height; i++)
    {
        if (fread(&(A->data[i]), 1, 1, fp) != 1)
            goto error_input_free;;
        if (fread(&(B->data[i]), 1, 1, fp) != 1)
            goto error_input_free;
        if (fread(&(C->data[i]), 1, 1, fp) != 1)
            goto error_input_free;
    }

    uint16_t d_height;
    uint16_t d_width;

    if (fread(&d_height, sizeof(d_height), 1, fp) != 1)
        goto error_input_free;

    if (fread(&d_width, sizeof(d_width), 1, fp) != 1)
        goto error_input_free;

    Matrix* D;
    if (matrix_alloc(&D, d_width, d_height) != ERROR_NONE)
    {
        matrix_free(&A);
        matrix_free(&B);
        matrix_free(&C);
        goto error_alloc;
    }

    if (fread(D->data, 1, d_width * d_height, fp) != d_width * d_height)
    {
        goto error_input_free_D;
    }

    *(res) = malloc(sizeof(Matrix*) * 4);
    if (*(res) == NULL)
    {
        matrix_free(&A);
        matrix_free(&B);
        matrix_free(&C);
        matrix_free(&D);
        goto error_alloc;
    }

    (*res)[0] = A;
    (*res)[1] = B;
    (*res)[2] = C;
    (*res)[3] = D;

    fclose(fp);
    return ERROR_NONE;

error_input_free_D:
    matrix_free(&D);

error_input_free:
    matrix_free(&A);
    matrix_free(&B);
    matrix_free(&C);

error_input:
    fclose(fp);
    return WRONG_INPUT;
error_alloc:
    fclose(fp);
    return MEMORY_ALLOCATION_ERROR;
}

/**
 * @brief Writes result matrices to a binary file
 * @param output Output file path
 * @param res Matrix array
 * @return Error code
 */
int write_output(const char* output, Matrix** res)
{
    FILE* fp = fopen(output, "wb+");
    if (fp == NULL) return WRONG_INPUT;
    if (fwrite(&(res[0]->height), sizeof(uint32_t), 1, fp) != 1)
    {
        goto error;
    }
    if (fwrite(&(res[0]->width), sizeof(uint32_t), 1, fp) != 1)
    {
        goto error;
    }
    for (uint32_t i = 0; i < res[0]->height * res[0]->width; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            if (fwrite(&((res)[j]->data[i]), sizeof(uint8_t), 1, fp) != 1) goto error;
        }
    }
    fclose(fp);
    return ERROR_NONE;
error:
    fclose(fp);
    return WRONG_INPUT;
}


/*
    as you said 5 params: prog, {("-i","input_path");("-o","outup_path")}
 */
int main(int argc, char* argv[])
{
    //printf("argc = %d\n", argc);

    if (argc < 5)
    {
        return WRONG_INPUT;
    }
    const char* input = NULL;
    const char* output = NULL;
    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "-o"))
        {
            if (i + 1 >= argc)
            {
                return WRONG_INPUT;
            }
            //printf("%s\n", argv[i+1]);
            output = argv[i + 1];
        }
        if (!strcmp(argv[i], "-i"))
        {
            if (i + 1 >= argc)
            {
                return WRONG_INPUT;
            }
            //printf("%s\n", argv[i+1]);
            input = argv[i + 1];
        }
    }

    if (input == NULL || output == NULL)
    {
        return WRONG_INPUT;
    }

    Matrix** res;
    if (read_input(input, &res))
    {
        return WRONG_INPUT;
    }
    // IMPORTANT!!!! A LOT!!!!
    // FOR NOT SPEND 1 SECOND MORE ON LARGE INPUT (by comuting rank) I MUTED THIS REALIZATION, UNCOMMENT FOR USE
    //int rank = rank_of(res[3]);
    int rank = -1;
    switch (rank)
    {
    case 0:
        fill_zero(res[0]);
        fill_zero(res[1]);
        fill_zero(res[2]);
        break;
    case 1:
        conversion_3_by_1(&res[0], &res[1], &res[2], res[3]);
        break;
    default:
        basic_3_conversion(&res[0], &res[1], &res[2], res[3]);
        break;
    }

    /**
    printf("A[0] = %u\n", res[0]->data[0]);
    printf("B[0] = %u\n", res[1]->data[0]);
    printf("C[0] = %u\n", res[2]->data[0]);
    **/
    if (write_output(output, res))
    {
        for (int i = 0; i < 4; i++) matrix_free(&res[i]);
        free(*res);
        return WRONG_INPUT;
    }

    for (int i = 0; i < 4; i++) matrix_free(&res[i]);
    free(*res);

    return ERROR_NONE;
}
