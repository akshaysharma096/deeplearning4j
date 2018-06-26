//
//  @author sgazeos@gmail.com
//
#ifndef __REDUCE_MIN_MAX_H_HELPERS__
#define __REDUCE_MIN_MAX_H_HELPERS__
#include <op_boilerplate.h>
#include <NDArray.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    void minMaxReduceFunctor(NDArray<T>* input, NDArray<T>* gradOut, NDArray<T>* tempVals, NDArray<T>* output, bool normalize = false);
//    void minMaxReduceFunctor(NDArray<T> const* input, NDArray<T> const* gradOut, NDArray<T> const* tempVals, NDArray<T>* output);

}
}
}
#endif //__REDUCE_MIN_MAX_H_HELPERS__

