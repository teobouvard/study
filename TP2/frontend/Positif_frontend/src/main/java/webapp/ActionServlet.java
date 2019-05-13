/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package webapp;

import serialisation.SerialisationVoyance;
import action.ActionChercherVoyance;
import serialisation.SerialisationEmploye;
import serialisation.SerialisationInfosMedium;
import action.ActionInfosMedium;
import serialisation.SerialisationListerMedium;
import action.ActionListerMediums;
import serialisation.SerialisationInscrire;
import action.Action;
import action.ActionInscrire;
import action.ActionLogin;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import dao.JpaUtil;
import java.io.IOException;
import java.io.PrintWriter;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;
import serialisation.Serialisation;
import serialisation.SerialisationClient;
import serialisation.SerialisationLogin;
import serialisation.SerialisationListeVoyance;

/**
 *
 * @author tbouvard
 */
@WebServlet(name = "ActionServlet", urlPatterns = {"/ActionServlet"})
public class ActionServlet extends HttpServlet {

    @Override
    public void init() throws ServletException {
        super.init();
        JpaUtil.init();
    }

    @Override
    public void destroy() {
        JpaUtil.destroy();
        super.destroy();
    }

    /**
     * Processes requests for both HTTP <code>GET</code> and <code>POST</code>
     * methods.
     *
     * @param request servlet request
     * @param response servlet response
     * @throws ServletException if a servlet-specific error occurs
     * @throws IOException if an I/O error occurs
     */
    protected void processRequest(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {

        String todo = request.getParameter("todo");
        Action action = null;
        Serialisation serialisation = null;
        HttpSession session = request.getSession();

        switch (todo) {
            case "connecter":
                action = new ActionLogin();
                serialisation = new SerialisationLogin();
                action.executer(request);
                serialisation.serialiser(request, response);
                break;

            case "inscrire":
                action = new ActionInscrire();
                serialisation = new SerialisationInscrire();
                action.executer(request);
                serialisation.serialiser(request, response);
                break;
            case "deconnecter":
                session.setAttribute("personne",null);
                break;
        }
        if (session.getAttribute("personne") != null) {
            switch (todo) {
                case "infos-client":
                    serialisation = new SerialisationClient();
                    serialisation.serialiser(request, response);
                    break;
                case "infos-employe":
                    serialisation = new SerialisationEmploye();
                    serialisation.serialiser(request, response);
                    break;
                case "liste-medium":
                    action = new ActionListerMediums();
                    serialisation = new SerialisationListerMedium();
                    action.executer(request);
                    serialisation.serialiser(request, response);
                    break;
                case "infos-medium":
                    action = new ActionInfosMedium();
                    serialisation = new SerialisationInfosMedium();
                    action.executer(request);
                    serialisation.serialiser(request, response);
                    break;
                case "historique-client":
                    serialisation = new SerialisationListeVoyance();
                    serialisation.serialiser(request, response);
                    break;
                case "voyance-en-cours":
                    action = new ActionChercherVoyance();
                    serialisation = new SerialisationVoyance();
                    action.executer(request);
                    serialisation.serialiser(request, response);
                    break;
            }
        }
    }

    // <editor-fold defaultstate="collapsed" desc="HttpServlet methods. Click on the + sign on the left to edit the code.">
    /**
     * Handles the HTTP <code>GET</code> method.
     *
     * @param request servlet request
     * @param response servlet response
     * @throws ServletException if a servlet-specific error occurs
     * @throws IOException if an I/O error occurs
     */
    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        processRequest(request, response);
    }

    /**
     * Handles the HTTP <code>POST</code> method.
     *
     * @param request servlet request
     * @param response servlet response
     * @throws ServletException if a servlet-specific error occurs
     * @throws IOException if an I/O error occurs
     */
    @Override
    protected void doPost(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        processRequest(request, response);
    }

    /**
     * Returns a short description of the servlet.
     *
     * @return a String containing servlet description
     */
    @Override
    public String getServletInfo() {
        return "Short description";
    }// </editor-fold>

}
