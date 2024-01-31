classdef ZScoreMap < DensityMetricBase
    %ZSCOREMAP Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        % Fields defined in base class
%         ImageHeight = 0;
%         ImageWidth = 0;
%         DensityMatrix = [];
        
        % for PCD
%         PCD_cppa = [];
%         MinDensity_cppa = 0;
%         PCD_loc = [];
        
        % for CDC
%         CDC20_density = 0;
%         CDC20_loc = [];
%         Stats2 = [];
        DensityName = [];
    end
    
    methods
        function obj = ZScoreMap(densityMap, CDC_loc, stats, PCD_loc, name)
            %ZSCOREMAP Construct an instance of this class
            %   Detailed explanation goes here
            obj.DensityMatrix = densityMap;
            [obj.ImageHeight, obj.ImageWidth, ~] = size(densityMap);
            obj.CDC20_loc = CDC_loc;
            obj.Stats2 = stats;
            obj.PCD_loc = PCD_loc;
            obj.PCD_cppa = densityMap(round(PCD_loc(2)), round(PCD_loc(1)));
            obj.CDC20_density = densityMap(round(CDC_loc(2)), round(CDC_loc(1)));
            obj.DensityName = name;
        end
        
        function Recalculate(obj)
        %   Recalculate(obj, sourceImage) 
            error("Nothing to calc")
        end

        function s = saveobj(obj)
            % for density map calculation
            s.ImageHeight = obj.ImageHeight;
            s.ImageWidth = obj.ImageWidth;
            s.DensityMatrix = obj.DensityMatrix;

            % for PCD
            s.PCD_cppa = obj.PCD_cppa;
            s.MinDensity_cppa = obj.MinDensity_cppa;
            s.PCD_loc = obj.PCD_loc;

            % for CDC
            s.CDC20_density = obj.CDC20_density;
            s.CDC20_loc = obj.CDC20_loc;
            s.Stats2 = obj.Stats2;
        end
    end

    methods(Static)
        function obj = loadobj(s)
            if isstruct(s)
                newObj = ZScoreMap(); 
                % for density map calculation
                newObj.ImageHeight = s.ImageHeight;
                newObj.ImageWidth = s.ImageWidth;
                newObj.DensityMatrix = s.DensityMatrix;

                % for PCD
                newObj.PCD_cppa = s.PCD_cppa;
                newObj.MinDensity_cppa = s.MinDensity_cppa;
                newObj.PCD_loc = s.PCD_loc;

                % for CDC
                newObj.CDC20_density = s.CDC20_density;
                newObj.CDC20_loc = s.CDC20_loc;
                newObj.Stats2 = s.Stats2;

                obj = newObj;
            else
                obj = s;
            end
        end


        function [densityMap] = GetDensityMatrix(densityMap)
            warning("nothing to calc")
        end
    end
end

