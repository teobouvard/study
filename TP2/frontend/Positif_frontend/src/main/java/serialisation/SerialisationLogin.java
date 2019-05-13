package serialisation;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import java.io.IOException;
import java.io.PrintWriter;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

public class SerialisationLogin extends Serialisation {

    @Override
    public void serialiser(HttpServletRequest request, HttpServletResponse response) throws IOException {
        PrintWriter out = getWriterWithJsonHeader(response);
        JsonObject jsonContainer = new JsonObject();
        Boolean statut = (Boolean) request.getAttribute("statut");
        
        if (statut) {
            jsonContainer.addProperty("connexion", true);
            jsonContainer.addProperty("message", "Ok");
            String typePersonne = (String) request.getAttribute("personne");
            if(typePersonne.equals("client")){
                jsonContainer.addProperty("personne", "client");
            }
            else if(typePersonne.equals("employe")){
                jsonContainer.addProperty("personne", "employe");
            }
        } else {
            jsonContainer.addProperty("connexion", false);
            jsonContainer.addProperty("message", "Mot de passe ou nom d'ultilisateur invalide.");
            
        }
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
            String json = gson.toJson(jsonContainer);
            out.println(json);
    }
}
