--[[ Usage:
    vector = require "vector"
    vec3 = vector(3)
    unit_x = vec3( 1.0, 0.0, 0.0 )
]]

local unpack = unpack or table.unpack
local pairs = pairs
local sin, cos, atan2 = math.sin, math.cos, math.atan2

--Reimplementation of moses.all but optimized for usage here
local function all ( t, f )
    local passed = true
    for i = 1, #t do
        passed = passed and f( i, t[i] )
    end
    return passed
end

--Reimplementation of moses.reduce but optimized for usage here
local function reduce ( t, f, state )
    local start_i = 1
    if not state then 
        state = t[1]
        start_i = 2
    end
    for i = start_i, #t do
        state = f( state, t[i] )
    end
    return state
end

local function rep ( value, n )
    local ret = {}
    for i = 1, n do
        ret[ i ] = value
    end
    return ret
end

--Creates a vector constructor for vectors of a given length
local function create_vec_library ( vec_length )
    assert( type(vec_length) == "number", "vec_length must be a number" )
    assert( vec_length ~= math.huge and vec_length == vec_length, "vec_length must be a finite number" )
    assert( vec_length % 1.0 == 0.0, "vec_length must be an integer" )
    assert( vec_length >= 1.0, "vec_length must be a positive value" )
    
    local meta = {}
    
    local method = {}
        meta.__index = method
    
    local function is_vec ( a )
        return getmetatable( a ) == meta
    end
    local function is_num ( a )
        return type( a ) == "number"
    end
    
    local function vector_constructor ( ... )
        local error_message = string.format( "argument must be either %i numbers or a(n) %i-element array of numbers", vec_length, vec_length )
        local args = { ... }
        
        assert( (#args == vec_length) or (#args == 1 and type(args[1] == "table") and #args[1] == vec_length), error_message )
        
        if #args == 1 then
            args = args[1]
        end
        
        assert( all( args, function ( _, element )
                return type(element) == "number"
            end ),
            error_message
        )
        
        return setmetatable( args, meta )
    end
    
    local static = setmetatable( {}, {
        __call = function ( self, ... ) return vector_constructor( ... ) end
    } )
    
    --[[ Calls function f( i, ... ), iterating the value of i over the range of the vector size.
    The returned values are used to initialize the value of a new vector, which is returned.]]
    local function create_vec_by_iter ( f, ... )
        local ret = {}
        for i = 1, vec_length do
            ret[i] = f( i, ... )
        end
        return setmetatable( ret, meta )
    end
    
    -- vec + vec
    do local function add_func ( i, l, r )
        return l[i] + r[i]
    end
    function meta.__add ( l, r )
        assert( is_vec(l) and is_vec(r), "can only add two same-size vectors together" )
        return create_vec_by_iter( add_func, l, r )
    end end
    
    -- vec - vec
    do local function sub_func ( i, l, r )
        return l[i] - r[i]
    end
    function meta.__sub ( l, r )
        assert( is_vec(l) and is_vec(r), "can only subtract two same-size vectors together" )
        return create_vec_by_iter( sub_func, l, r )
    end end
    
    -- vec * num
    -- num * vec
    do local function mul_func ( i, vec, scal )
        return vec[i] * scal
    end
    function meta.__mul ( l, r )
        assert( (is_vec(l) and is_num(r))
             or (is_num(l) and is_vec(r)),
             "can only multiply a vector with a scalar" )
        
        local vec, scal
        if is_vec(l) then
            vec, scal = l, r
        else
            vec, scal = r, l
        end
        
        return create_vec_by_iter( mul_func, vec, scal )
    end end
    
    -- vec / num
    do local function div_func ( i, vec, scal )
        return vec[i] / scal
    end
    function meta.__div ( l, r )
        assert( is_vec(l) and is_num(r), "can only divide a vector by a scalar" )
        return create_vec_by_iter( div_func, l, r )
    end end
    
    -- vec ^ num
    do local function pow_func ( i, vec, scal )
        return vec[i]^scal
    end
    function meta.__pow ( l, r )
        assert( is_vec(l) and is_num(r), "can only exponentiate a vector by a scalar" )
        return create_vec_by_iter( pow_func, l, r )
    end end
    
    -- -vec
    do local function unm_func ( i, vec )
        return -vec[i]
    end
    function meta:__unm ( )
        return create_vec_by_iter( unm_func, self )
    end end
    
    -- vec == vec
    do local function eq_func( i, _, l, r )
        return l[i] == r[i]
    end
    function meta.__eq ( l, r )
        if not (is_vec(l) and is_vec(r)) then
            return false
        else
            return all( l, eq_func, l, r )
        end
    end end
    
    -- tostring( vec )
    function meta:__tostring ( )
        return self:tostring()
    end
    
    -- vec:tostring( [fmt] )
    --[[function method:tostring( fmt )
        return "(" .. table.concat( _.map( self, function(_,v)
            return (fmt or "%g"):format( v )
        end ), ", " ) .. ")"
    end--]]
    function method:tostring( fmt )
        local val_strings = {}
        for i,v in ipairs( self ) do
            val_strings[i] = (fmt or "%g"):format( v )
        end
        return "(" .. table.concat( val_strings, "," ) .. ")"
    end
    
    -- vec:abs()
    do
        local abs = math.abs
        local function abs_func ( i, vec )
            return abs( vec[i] )
        end
        function method:abs ( )
            return create_vec_by_iter( abs_func, self )
        end
    end
    
    -- vec:len()
    do 
        local sqrt = math.sqrt
        local function len_func ( state, element )
            return state + element^2
        end
        function method:len ( )
            return sqrt( reduce( self, len_func, 0.0 ) )
        end
    end
    
    -- vec:dot( vec )
    function method:dot ( r )
        local l = self
        assert( is_vec( r ), "can only calculate dot-product between two equal-size vectors" )
        
        local ret = 0.0
        for i = 1, vec_length do
            ret = ret + l[i]*r[i]
        end
        
        return ret
    end
    
    -- vec:cross( vec )
    -- Only valid for 3-dimensional vectors
    if vec_length == 3 then
        function method:cross ( r )
            local l = self
            assert( is_vec(r), "can only calculate cross-product between two 3-dimensional vectors" )
            return setmetatable( {
                l[2]*r[3] - r[2]*l[3],
                l[3]*r[1] - r[3]*l[1],
                l[1]*r[2] - r[1]*l[2]
            }, meta )
        end
    end
    
    -- vec:norm()
    function method:norm ( )
        return self / self:len()
    end
    
    -- vec:rotate( ang )
    -- This form only valid for 2-dimensional vectors, other forms may be implemented later
    if vec_length == 2 then
        local whole_rev = 2*math.pi
        function method:rotate ( ang )
            assert( type(ang) == "number" and (-whole_rev < ang) and (ang < whole_rev), "argument must be a number less than a full revolution" )
            local x1, y1 = self[1], self[2]
            
            local x2 = cos(ang)*x1 - sin(ang)*y1
            local y2 = sin(ang)*x1 + cos(ang)*y1
            
            return vector_constructor ( x2, y2 )
        end
    end
    
    -- vec:ang()
    -- Only valid for 2-dimensional vectors
    if vec_length == 2 then
        function method:ang ( )
            return atan2( self[2], self[1] )
        end
    end
    
    -- vec(3+):truncate( n )
    if vec_length >= 3 then
        function method:truncate ( n )
            assert( type(n) == "number", "n must be a number" )
            assert( (n % 1.0 == 0.0) and (n >= 2) and (n < vec_length), "n must be an integer at least 2 and less than the length of the original vector" )
            
            local new_vec_create = create_vec_library( n )
            local ret = {}
            
            for i = 1, n do
                ret[i] = self[i]
            end
            
            return new_vec_create( unpack( ret ) )
        end
    end
    
    -- vec(n):extend( n )
    function method:extend ( n )
        assert( type(n) == "number", "n must be a number" )
        assert( (n % 1.0 == 0.0) and (n > vec_length), "n must be an integer greater than the length of the original vector" )
        
        local new_vec_create = create_vec_library( n )
        local ret = {}
        
        for i = 1, vec_length do
            ret[i] = self[i]
        end
        
        for i = vec_length+1, n do
            ret[i] = 0.0
        end
        
        return new_vec_create( unpack( ret ) )
    end
    
    -- static.all( num )
    function static.all( num )
        assert( type(num) == "number", "only accepts number argument" )
        return setmetatable( rep( num, vec_length ), meta )
    end
    
    -- static.unit( axis )
    function static.unit( axis_n )
        assert( (type(axis_n) == "number") and (axis_n % 1.0 == 0.0) and (1.0 <= axis_n and axis_n <= vec_length ),
            string.format( "only accepts positive integer between 1 and %i", vec_length )
        )
        local ret = rep( 0.0, vec_length )
        ret[ axis_n ] = 1.0
        return setmetatable( ret, meta )
    end
    
    -- static.circ( ang, len )
    -- Only valid for 2-dimensional vectors
    if vec_length == 2 then
        function static.circ( ang, len )
            assert( (type(ang) == "number") and (type(len) == "number"), "only accepts numbers for both arguments" )
            return vector_constructor( cos(ang)*len, sin(ang)*len )
        end
    end
    
    return static
end

local created_libraries = {}
return function ( vec_length )
    local ret = created_libraries[ vec_length ]
    if not ret then
        ret = create_vec_library( vec_length )
        created_libraries[ vec_length ] = ret
    end
    return ret
end