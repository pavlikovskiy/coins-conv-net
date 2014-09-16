%% Predict CountryId by CIID using exported file from BIND_SUDGEST
function [countryId] = predictCountryFromFile(ciid, ciidCountriesMap)

    countryId = '-1';
    for j = 1 : size(ciidCountriesMap{1}, 1)
        if strcmp(ciid, ciidCountriesMap{1}{j})
            countryId = ciidCountriesMap{2}{j};
            break;
        end
    end
    
end