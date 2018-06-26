//
// Created by raver119 on 29/10/17.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_reshape)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        //////////////////////////////////////////////////////////////////////////
        // here iArgs is a vector with (optional) negative of order as first element:
        // ({-order, dim1, dim2, dim3, ...})
        CUSTOM_OP_IMPL(reshape, 1, 1, true, 0, -2) {
            auto x = INPUT_VARIABLE(0);

            if (block.width() == 1) {
                auto arguments = block.getIArguments();
                int argsSize = arguments->size();

                int e = 1;
                char order = (char) -(*arguments)[0];
                if (order != 'c' && order != 'f') {
                    order = x->ordering();
                    e = 0;
                }

                REQUIRE_TRUE(argsSize - e >= 1, 0, "Reshape arguments should at least 1 dimension");

                std::vector<Nd4jLong> shapeNew;
                
                for (; e < (int) arguments->size(); e++)
                    shapeNew.push_back((int) arguments->at(e));

                auto len = shape::prodLong(shapeNew.data(), shapeNew.size());
                REQUIRE_TRUE(len == x->lengthOf(), 0, "Reshape: lengths before and after reshape should match, but got %i vs %i", x->lengthOf(), len);

                if (Environment::getInstance()->isDebugAndVerbose()) {
                    nd4j_printv("Reshape: new shape", shapeNew);
                }

                if (block.isInplace()) {
                    if (x->reshapei(order, shapeNew)) {
                        STORE_RESULT(*x);
                        return ND4J_STATUS_OK;
                    }
                } else {
                    auto ret = OUTPUT_VARIABLE(0);
                    auto xr = x->reshape(order, shapeNew);
                    ret->assign(xr);
                    STORE_RESULT(*ret);
                    delete xr;
                    return ND4J_STATUS_OK;
                }
            } else if (block.width() == 2) {
                auto s = INPUT_VARIABLE(1);

                char order = 'c';
                if (block.numI() > 0)
                    order = (char) -INT_ARG(0);

                std::vector<Nd4jLong> shapeNew(s->lengthOf());

                for (int e = 0; e < (int) s->lengthOf(); e++)
                    shapeNew[e] = static_cast<Nd4jLong>(s->getScalar(e));

               if (Environment::getInstance()->isDebugAndVerbose()) {
                    nd4j_printv("Reshape: new shape", shapeNew);
                }

                if (block.isInplace()) {
                    if (x->reshapei(order, shapeNew)) {
                        STORE_RESULT(*x);
                        return Status::OK();
                    }
                } else {
                    auto ret = OUTPUT_VARIABLE(0);
                    if (s->isEmpty()) {
                        // just a scalar
                        ret->assign(x);
                    } else {
                        auto xr = x->reshape(order, shapeNew);
                        ret->assign(xr);
                        delete xr;
                    }

                    return Status::OK();
                }
            }

            return ND4J_STATUS_BAD_INPUT;
        }
        DECLARE_SHAPE_FN(reshape) {
            auto inp = inputShape->at(0);

            // we can launch op using Int arguments
            if (inputShape->size() == 1) {
                std::vector<int> *arguments = block.getIArguments();

                int e = 1;
                char order = (char) -(*arguments)[0];
                if (order != 'c' && order != 'f') {
                    order = shape::order(inp);
                    e = 0;
                }

                std::vector<int> shapeNew;

                for (; e < (int) arguments->size(); e++)
                    shapeNew.push_back((int) arguments->at(e));

                Nd4jLong *newShape;
                ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength((int) shapeNew.size()), Nd4jLong);

                int numberNegativesOnes = 0;

                int *shape_ = shapeNew.data();
                for (int i = 0; i < (int) shapeNew.size(); i++) {
                    if (shapeNew[i] == -1) {
                        if (numberNegativesOnes >= 1)
                            throw std::runtime_error("Only one dimension can be negative ones");

                        numberNegativesOnes++;

                        int shapeLength = 1;
                        for (int j = 0; j < (int) shapeNew.size(); j++)
                            if (shape_[j] >= 1)
                                shapeLength *= shape_[j];

                        // FIXME: use workspace here
                        int realShape = nd4j::math::nd4j_abs<int>((int) shape::length(inp) / shapeLength);
                        int *thisNewShape = new int[shapeNew.size()];

                        for (int j = 0; j < (int) shapeNew.size(); j++) {
                            if (i != j) {
                                thisNewShape[j] = shape_[j];
                            } else
                                thisNewShape[j] = realShape;
                        }

                        shape_ = thisNewShape;
                        break;
                    }
                }

                for (int f = 0; e < (int) shapeNew.size(); f++) {
                    shapeNew[f] = shape_[f];
                }

                if (numberNegativesOnes > 0)
                    delete[] shape_;

                newShape[0] = shapeNew.size();
                int cnt = 1;
                for (auto v: shapeNew)
                    newShape[cnt++] = v;

                shape::updateStrides(newShape, order);

                return SHAPELIST(newShape);
            } else {
                // or, with second input "as shape"
                auto x = INPUT_VARIABLE(0);
                auto y = INPUT_VARIABLE(1);

                // special case here
                if (y->isEmpty()) {
                    REQUIRE_TRUE(x->lengthOf() == 1, 0, "Reshape: new length doesn't match existing array");


                    return SHAPELIST(ShapeUtils<T>::createScalarShapeInfo(block.getWorkspace()));
                }

                std::vector<Nd4jLong> shapeNew(y->lengthOf());
                int numberNegativesOnes = 0;

                for (int e = 0; e < (int) y->lengthOf(); e++)
                    shapeNew[e] = (int) y->getIndexedScalar(e);

                auto shape_ = shapeNew.data();
                for (int i = 0; i < (int) shapeNew.size(); i++) {
                    if (shapeNew[i] == -1) {
                        if (numberNegativesOnes >= 1)
                            throw std::runtime_error("Only one dimension can be negative ones");

                        numberNegativesOnes++;

                        int shapeLength = 1;
                        for (int j = 0; j < (int) shapeNew.size(); j++)
                            if (shape_[j] >= 1)
                                shapeLength *= shape_[j];

                        // FIXME: use workspace here
                        auto realShape = nd4j::math::nd4j_abs<Nd4jLong>(x->lengthOf()/ shapeLength);
                        auto thisNewShape = new Nd4jLong[shapeNew.size()];

                        for (int j = 0; j < (int) shapeNew.size(); j++) {
                            if (i != j) {
                                thisNewShape[j] = shape_[j];
                            } else
                                thisNewShape[j] = realShape;
                        }

                        shape_ = thisNewShape;
                        break;
                    }
                }

                for (int e = 0; e < (int) shapeNew.size(); e++) {
                    shapeNew[e] = shape_[e];
                }

                if (numberNegativesOnes > 0)
                    delete[] shape_;

                Nd4jLong *newShape;
                ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(shapeNew.size()), Nd4jLong);

                shape::shapeBuffer(shapeNew.size(), shapeNew.data(), newShape);

                return SHAPELIST(newShape);
            }
        }
    }
}

#endif