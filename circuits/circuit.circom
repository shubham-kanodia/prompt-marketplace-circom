pragma circom 2.0.0;

include "../node_modules/circomlib-ml/circuits/circomlib-matrix/matElemMul.circom";
include "../node_modules/circomlib-ml/circuits/circomlib-matrix/matElemSum.circom";
include "../node_modules/circomlib-ml/circuits/util.circom";

template Conv2D (nRows, nCols, nChannels, nFilters, kernelSize, strides) {
    signal input in[nRows][nCols][nChannels];
    signal output out[(nRows-kernelSize)\strides+1][(nCols-kernelSize)\strides+1][nFilters];

    signal input weights[kernelSize][kernelSize][nChannels][nFilters];    
    signal input bias[nFilters];

    component mul[(nRows-kernelSize)\strides+1][(nCols-kernelSize)\strides+1][nChannels][nFilters];
    component elemSum[(nRows-kernelSize)\strides+1][(nCols-kernelSize)\strides+1][nChannels][nFilters];
    component sum[(nRows-kernelSize)\strides+1][(nCols-kernelSize)\strides+1][nFilters];

    for (var i=0; i<(nRows-kernelSize)\strides+1; i++) {
        for (var j=0; j<(nCols-kernelSize)\strides+1; j++) {
            for (var k=0; k<nChannels; k++) {
                for (var m=0; m<nFilters; m++) {
                    mul[i][j][k][m] = parallel matElemMul(kernelSize,kernelSize);
                    for (var x=0; x<kernelSize; x++) {
                        for (var y=0; y<kernelSize; y++) {
                            mul[i][j][k][m].a[x][y] <== in[i*strides+x][j*strides+y][k];
                            mul[i][j][k][m].b[x][y] <== weights[x][y][k][m];
                        }
                    }
                    elemSum[i][j][k][m] = parallel matElemSum(kernelSize,kernelSize);
                    for (var x=0; x<kernelSize; x++) {
                        for (var y=0; y<kernelSize; y++) {
                            elemSum[i][j][k][m].a[x][y] <== mul[i][j][k][m].out[x][y];
                        }
                    }
                }
            }
            for (var m=0; m<nFilters; m++) {
                sum[i][j][m] = Sum(nChannels);
                for (var k=0; k<nChannels; k++) {
                    sum[i][j][m].in[k] <== elemSum[i][j][k][m].out;
                }
                out[i][j][m] <== sum[i][j][m].out + bias[m];
            }
        }
    }
}

component main = Conv2D(64, 64, 32, 3, 3, 1);