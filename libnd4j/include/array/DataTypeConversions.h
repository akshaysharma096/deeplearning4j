//
// Created by raver119 on 21.11.17.
//

#ifndef LIBND4J_DATATYPECONVERSIONS_H
#define LIBND4J_DATATYPECONVERSIONS_H

#include <pointercast.h>
#include <helpers/logger.h>
#include <op_boilerplate.h>
#include <array/DataType.h>
#include <types/float16.h>
#include <helpers/BitwiseUtils.h>

namespace nd4j {
    template <typename T>
    class DataTypeConversions {
    public:
        static FORCEINLINE void convertType(T* buffer, void* src, DataType dataType, ByteOrder order, Nd4jLong length) {
            bool isBe = BitwiseUtils::isBE();
            bool canKeep = (isBe && order == ByteOrder::BE) || (!isBe && order == ByteOrder::LE);

            switch (dataType) {
                case DataType_FLOAT: {
                        if (std::is_same<T, float>::value && canKeep) {
                            memcpy(buffer, src, length * sizeof(T));
                        } else {
                            auto tmp = reinterpret_cast<float *>(src);

#if __GNUC__ <= 4
                            if (!canKeep)
                                for (Nd4jLong e = 0; e < length; e++)
                                    buffer[e] = BitwiseUtils::swap_bytes<T>(static_cast<T>(tmp[e]));
                            else
                                for (Nd4jLong e = 0; e < length; e++)
                                    buffer[e] = static_cast<T>(tmp[e]);
#else
                            //#pragma omp parallel for simd schedule(guided)
                            for (Nd4jLong e = 0; e < length; e++)
                                buffer[e] = canKeep ? static_cast<T>(tmp[e]) : BitwiseUtils::swap_bytes<T>(static_cast<T>(tmp[e]));
#endif
                        }
                    }
                    break;
                case DataType_DOUBLE: {
                        if (std::is_same<T, double>::value && canKeep) {
                            memcpy(buffer, src, length * sizeof(T));
                        } else {
                            auto tmp = reinterpret_cast<double *>(src);

#if __GNUC__ <= 4
                            if (!canKeep)
                                for (Nd4jLong e = 0; e < length; e++)
                                    buffer[e] = BitwiseUtils::swap_bytes<T>(static_cast<T>(tmp[e]));
                            else
                                for (Nd4jLong e = 0; e < length; e++)
                                    buffer[e] = static_cast<T>(tmp[e]);
#else
                            //#pragma omp parallel for simd schedule(guided)
                            for (Nd4jLong e = 0; e < length; e++)
                                buffer[e] = canKeep ? static_cast<T>(tmp[e]) : BitwiseUtils::swap_bytes<T>(static_cast<T>(tmp[e]));
#endif
                        }
                    }
                    break;
                case DataType_HALF: {

                        if (std::is_same<T, float16>::value && canKeep) {
                            memcpy(buffer, src, length * sizeof(T));
                        } else {
                            auto tmp = reinterpret_cast<float16 *>(src);

#if __GNUC__ <= 4
                            if (!canKeep)
                                for (Nd4jLong e = 0; e < length; e++)
                                    buffer[e] = BitwiseUtils::swap_bytes<T>(static_cast<T>(tmp[e]));
                            else
                                for (Nd4jLong e = 0; e < length; e++)
                                    buffer[e] = static_cast<T>(tmp[e]);
#else
                            //#pragma omp parallel for simd schedule(guided)
                            for (Nd4jLong e = 0; e < length; e++)
                                buffer[e] = canKeep ? static_cast<T>(tmp[e]) : BitwiseUtils::swap_bytes<T>(static_cast<T>(tmp[e]));
#endif
                        }
                    }
                    break;
                default: {
                    nd4j_printf("Unsupported DataType requested: [%i]\n", static_cast<int>(dataType));
                    throw std::runtime_error("Unsupported DataType");
                }
            }
        }
    };
}



#endif //LIBND4J_DATATYPECONVERSIONS_H
