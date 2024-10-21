function out = ThousandSep(in)
%THOUSANDSEP adds thousands Separators to a 1x1 array.
%   Example:
%      ThousandSep(1234567)
    import java.text.*
    v = DecimalFormat;
    out = char(v.format(in));
end