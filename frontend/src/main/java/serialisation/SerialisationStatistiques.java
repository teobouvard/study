package serialisation;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import metier.data.Employe;
import metier.data.Medium;
import static metier.service.Service.getListEmploye;
import static metier.service.Service.getListMedium;

public class SerialisationStatistiques extends Serialisation {

    @Override
    public void serialiser(HttpServletRequest request, HttpServletResponse response) throws IOException {
        PrintWriter out = getWriterWithJsonHeader(response);
        JsonObject jsonContainer = new JsonObject();

        List<Medium> listeMedium = (List<Medium>) request.getAttribute("listeMedium");
        List<Employe> listeEmploye = (List<Employe>) request.getAttribute("listeEmploye");

        JsonArray listeMediumJson = new JsonArray();
        Integer nbVoyancesTotal = 0;
        for (Medium medium : listeMedium) {
            JsonObject mediumJson = new JsonObject();
            mediumJson.addProperty("nom", medium.getNom());
            mediumJson.addProperty("nbVoyances", medium.getListVoyance().size());
            nbVoyancesTotal += medium.getListVoyance().size();
            listeMediumJson.add(mediumJson);
        }
        jsonContainer.add("listeMedium", listeMediumJson);
        jsonContainer.addProperty("nbVoyancesTotal", nbVoyancesTotal);

        JsonArray listeEmployeJson = new JsonArray();
        Integer nbSessionsTotal = 0;
        for (Employe employe : listeEmploye) {
            JsonObject employeJson = new JsonObject();
            employeJson.addProperty("nom", employe.getNom());
            employeJson.addProperty("prenom", employe.getPrenom());
            employeJson.addProperty("nbSessions", employe.getListVoyance().size());
            nbSessionsTotal += employe.getListVoyance().size();
            listeEmployeJson.add(employeJson);
        }
        jsonContainer.add("listeEmploye", listeEmployeJson);
        jsonContainer.addProperty("nbSessionsTotal", nbSessionsTotal);

        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        String json = gson.toJson(jsonContainer);
        out.println(json);
    }

}
