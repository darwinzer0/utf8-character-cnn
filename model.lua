assert(pcall(function() require('dpnn') end), 'dpnn module required: luarocks install dpnn')

-- return function that returns network definition
return function(params)
    -- get number of classes from external parameters (default to 14)
    local nclasses = params.nclasses or 2

    if pcall(function() require('cudnn') end) then
       print('Using CuDNN backend')
       backend = cudnn
       convLayer = cudnn.SpatialConvolution
       convLayerName = 'cudnn.SpatialConvolution'
    else
       print('Failed to load cudnn backend (is libcudnn.so in your library path?)')
       if pcall(function() require('cunn') end) then
           print('Falling back to legacy cunn backend')
       else
           print('Failed to load cunn backend (is CUDA installed?)')
           print('Falling back to legacy nn backend')
       end
       backend = nn -- works with cunn or nn
       convLayer = nn.SpatialConvolutionMM
       convLayerName = 'nn.SpatialConvolutionMM'
    end

    local feature_len = 1
    if params.inputShape then
        assert(params.inputShape[1]==1, 'Network expects 1xHxW images')
        params.inputShape:apply(function(x) feature_len=feature_len*x end)
    end

    local alphabet_len = 256 -- max index in input samples

    local net = nn.Sequential()
    -- feature_len x 1 x 1
    net:add(nn.View(-1,feature_len))
    -- feature_len
    net:add(nn.OneHot(alphabet_len))
    -- feature_len x alphabet_len
    net:add(backend.TemporalConvolution(alphabet_len, 256, 7))
    -- [(576-6)=570] x 256
    net:add(nn.Threshold())
    net:add(nn.TemporalMaxPooling(3, 3))
    -- [(570-3)/3+1=190] x 256
    net:add(backend.TemporalConvolution(256, 256, 7))
    -- [190-6=184] x 256
    net:add(nn.Threshold())
    net:add(nn.TemporalMaxPooling(3, 3))
    -- [(184-3)/3+1=61] x 256
    net:add(backend.TemporalConvolution(256, 256, 3))
    net:add(nn.Threshold())
    -- [61-2=59] x 256
    net:add(backend.TemporalConvolution(256, 256, 3))
    net:add(nn.Threshold())
    -- [59-2=57] x 256
    net:add(backend.TemporalConvolution(256, 256, 3))
    net:add(nn.Threshold())
    -- [57-2=55] x 256
    net:add(backend.TemporalConvolution(256, 256, 3))
    net:add(nn.Threshold())
    -- [55-2=53] x 256
    net:add(nn.TemporalMaxPooling(3, 3))
    -- [(53-3)/3+1=17] x 256
    net:add(nn.Reshape(4352))
    -- 4352
    net:add(nn.Linear(4352, 1024))
    net:add(nn.Threshold())
    net:add(nn.Dropout(0.5))
    -- 1024
    net:add(nn.Linear(1024, 1024))
    net:add(nn.Threshold())
    net:add(nn.Dropout(0.5))
    -- 1024
    net:add(nn.Linear(1024, nclasses))
    net:add(backend.LogSoftMax())

    -- weight initialization
    local w,dw = net:getParameters()
    w:normal():mul(5e-2)

    return {
        model = net,
        loss = nn.ClassNLLCriterion(),
        trainBatchSize = 128,
        validationBatchSize = 128
    }
end
